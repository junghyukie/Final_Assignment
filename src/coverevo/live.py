"""Direct, budgeted LIVE discovery; separate pilot explicitly skipped by the user."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import random
import re
from typing import Callable

import requests
import yaml

from .budget import BudgetLedger, FileLock, RequestLifecycle, validate_live_budget
from .contracts import ExperimentConfig, ModelConfig
from .instances import build_datasets, derive_seed, serialize_dataset_manifest
from .openrouter_client import ManagedOpenRouterProvider, OpenRouterClient
from .protocol import _tree_hash
from .search import SearchStoppedError, run_search, run_state_from_mapping, run_state_to_mapping
from .storage import dump_json_atomic, encode_lossless, load_json_lossless


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(encode_lossless(value), sort_keys=True,
        separators=(',', ':')).encode('utf-8')).hexdigest()


def _decimal(value, label: str, *, positive: bool = False) -> Decimal:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(label + ' must be a finite USD value') from None
    if not number.is_finite() or number < 0 or (positive and number == 0):
        raise ValueError(label + ' must be a finite non-negative USD value')
    return number


def read_live_config(path: str | Path) -> ExperimentConfig:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))
    except yaml.YAMLError:
        raise ValueError('invalid live YAML configuration') from None
    def reject_secrets(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower() in {'api_key', 'openrouter_api_key', 'authorization'}:
                    raise ValueError('credentials belong in .env, not YAML')
                reject_secrets(item)
        elif isinstance(value, list):
            for item in value:
                reject_secrets(item)
    reject_secrets(raw)
    if not isinstance(raw, dict):
        raise ValueError('live config must be a YAML mapping')
    config = ExperimentConfig.from_mapping(raw)
    if config.mode != 'LIVE' or config.allow_live is not True:
        raise ValueError('config must set mode=LIVE and allow_live=true')
    if not re.fullmatch(r'[\w.:-]+/[\w.:-]+', str(config.model.get('id', ''))):
        raise ValueError('model.id must be an explicit author/model slug')
    validate_live_budget(True,
        _decimal(config.budget.get('initial_verified_allowance_usd'), 'configured allowance', positive=True),
        _decimal(config.budget.get('local_cap_usd'), 'local cap', positive=True))
    methods = config.search.get('methods')
    if (not isinstance(methods, list) or not methods or len(set(methods)) != len(methods)
        or set(methods) - {'IND', 'EVO_P', 'EVO_F', 'RS_PARAM'}):
        raise ValueError('search.methods must contain unique supported methods')
    for field in ('runs', 'slots_per_run', 'archive_size', 'initial_fresh_slots', 'mutation_fresh_period'):
        if type(config.search.get(field)) is not int or config.search[field] <= 0:
            raise ValueError('search.' + field + ' must be a positive integer')
    if config.search.get('auto_post_retries', 0) != 0 or config.budget.get('concurrent_requests', 1) != 1:
        raise ValueError('LIVE requires zero POST retries and one concurrent request')
    # Validate the payload without reading credentials or making HTTP requests.
    model = ModelConfig.from_mapping(config.model)
    OpenRouterClient._validate_config(replace(model,
        canonical_slug_verified=model.canonical_slug_verified or 'pending/catalog',
        provider_slug_verified=model.provider_slug_verified or 'pending-provider'))
    return config


def load_api_key(env_path: str | Path) -> str:
    """Read just one dotenv variable, without executing or logging file contents."""
    key = os.environ.get('OPENROUTER_API_KEY', '').strip()
    if not key:
        path = Path(env_path)
        if path.is_file():
            for line in path.read_text(encoding='utf-8-sig').splitlines():
                text = line.strip()
                if text.startswith('export '):
                    text = text[7:].lstrip()
                name, separator, value = text.partition('=')
                if separator and name.strip() == 'OPENROUTER_API_KEY':
                    key = value.strip()
                    if len(key) >= 2 and key[0] in {'"', "'"} and key[-1] == key[0]:
                        key = key[1:-1]
                    else:
                        key = key.split(' #', 1)[0].strip()
    if not key or any(char.isspace() for char in key):
        raise ValueError('set a nonempty OPENROUTER_API_KEY in .env or the environment')
    return key


def _prices(endpoint: dict) -> dict[str, Decimal]:
    pricing = endpoint.get('pricing') or {}
    result = {name: _decimal(pricing.get(name), 'endpoint pricing.'+name)
              for name in ('prompt', 'completion')}
    result['request'] = _decimal(pricing.get('request', '0'), 'endpoint pricing.request')
    for name, value in pricing.items():
        if name in {'image', 'audio', 'input_audio_cache', 'web_search'}:
            # The fixed payload contains text only and never enables search/tools.
            continue
        if name == 'internal_reasoning':
            # Reasoning shares the configured output cap. Reserve at the higher rate.
            result['completion'] = max(result['completion'], _decimal(value, 'reasoning pricing'))
            continue
        if name not in {'prompt', 'completion', 'request', 'input_cache_read', 'input_cache_write'}:
            if _decimal(value, 'additional endpoint pricing') != 0:
                raise ValueError('unsupported nonzero endpoint pricing unit')
        elif name in {'input_cache_read', 'input_cache_write'}:
            # Reserve the highest input-token rate, including cache writes.
            result['prompt'] = max(result['prompt'], _decimal(value, 'cache pricing'))
    return result


def _catalog(client, config, preferred_provider=None):
    try:
        key = client.get_key_info()
        model = client.get_model(config.model['id'])
        endpoints = client.get_endpoints(config.model['id'])
    except requests.RequestException:
        raise RuntimeError('OpenRouter catalog/key request failed; no generation sent') from None
    # Zero blocks a new POST, but must not block reconciliation of saved responses.
    remaining = _decimal(key.get('limit_remaining'), 'verified key limit_remaining')
    if model.get('id') != config.model['id'] or not model.get('canonical_slug'):
        raise ValueError('model catalog did not resolve the configured model')
    explicit_canonical = config.model.get('canonical_slug_verified')
    if explicit_canonical and explicit_canonical != model['canonical_slug']:
        raise ValueError('configured canonical slug differs from catalog')
    required = {'temperature', 'max_tokens', 'response_format', 'reasoning'}
    if required - set(model.get('supported_parameters') or []):
        raise ValueError('model does not support all configured parameters')
    provider = preferred_provider or config.model.get('provider_slug_verified')
    candidates = []
    for endpoint in endpoints:
        if (not endpoint.get('tag') or endpoint.get('status') != 0
            or (provider and endpoint['tag'] != provider)
            or required - set(endpoint.get('supported_parameters') or [])):
            continue
        maximum = endpoint.get('max_completion_tokens')
        if isinstance(maximum, int) and maximum < config.model['max_tokens']:
            continue
        try:
            prices = _prices(endpoint)
        except ValueError:
            continue
        candidates.append((prices['prompt']+prices['completion'], endpoint['tag'], endpoint))
    if not candidates:
        raise ValueError('no compatible provider endpoint with supported finite pricing')
    endpoint = sorted(candidates, key=lambda entry: entry[:2])[0][2]
    return key, model, endpoint, remaining


class _ReservedProvider(ManagedOpenRouterProvider):
    def propose(self, messages, config):
        artifact = self.artifact_directory/f'slot-{self.slot:04d}.json'
        if self.lifecycle.state is not None and self.lifecycle.state.get('slot') == self.slot and artifact.is_file():
            # This path only verifies/reconciles a durable response; it sends no POST.
            return super().propose(messages, config)
        _, model, endpoint, allowance = _catalog(self.client, config, self.endpoint['tag'])
        if {'model': model, 'endpoint': endpoint} != self.catalog:
            raise ValueError('model/provider catalog or pricing changed before the next request')
        prices = _prices(self.endpoint)
        # A conservative UTF-8-byte estimate plus chat framing, not a measured tokenizer bound.
        input_estimate = len(json.dumps(messages, ensure_ascii=False).encode('utf-8')) + 1024
        maximum = self.model_context
        if isinstance(maximum, int) and input_estimate + config.model['max_tokens'] > maximum:
            raise ValueError('estimated request exceeds catalog context length')
        self.estimate = max(Decimal('0.00000001'), Decimal('1.5') * (
            input_estimate * prices['prompt'] + config.model['max_tokens'] * prices['completion']
            + prices['request']))
        if self.estimate > allowance:
            raise ValueError('request reservation exceeds current key allowance')
        return super().propose(messages, config)


def run_live_discovery(config_path, output, *, env_path='.env', allow_live=False,
                       skip_pilot=False, max_attempts=None, client_factory=None,
                       progress: Callable[[dict], None] | None = None) -> dict:
    """Run real discovery only. Selection/test and a main-protocol freeze are separate."""
    if not allow_live:
        raise ValueError('real requests require --allow-live')
    if not skip_pilot:
        raise ValueError('direct discovery requires explicit --skip-pilot; no pilot evidence is fabricated')
    if output is None:
        raise ValueError('LIVE requires an explicit --output directory')
    if max_attempts is not None and (type(max_attempts) is not int or max_attempts < 0):
        raise ValueError('max-attempts must be a non-negative integer')
    config = read_live_config(config_path)
    source = Path(__file__).parent
    train = build_datasets(config, splits=('train',))
    dataset = json.loads(serialize_dataset_manifest(train))
    identity = {
        'config': hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        'source': _tree_hash(source, ('.py',)),
        'prompts': _tree_hash(source.parents[1]/'prompts', ('.txt',)),
        'dataset': _digest(dataset),
        'protocol_deviations': ['PILOT_SKIPPED_BY_USER'],
    }
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(root/'run.lock'):
        runtime_path = root/'runtime.json'
        runtime = load_json_lossless(runtime_path) if runtime_path.exists() else None
        if runtime is not None:
            if runtime.get('identity') != identity:
                raise ValueError('LIVE input/source/prompt identity changed; use a new output directory')
            if load_json_lossless(root/'dataset.json') != dataset:
                raise ValueError('saved training dataset changed')
            if _digest(load_json_lossless(root/'catalog.json')) != runtime['catalog_hash']:
                raise ValueError('saved catalog changed')
        elif set(item.name for item in root.iterdir()) - {'run.lock'}:
            raise ValueError('output is nonempty and has no LIVE runtime record')

        checkpoint_path = root/'discovery.json'
        checkpoint = load_json_lossless(checkpoint_path) if checkpoint_path.exists() else None
        if checkpoint is not None:
            if checkpoint.get('artifact_class') != 'LIVE' or checkpoint.get('input_hash') != _digest(runtime):
                raise ValueError('LIVE checkpoint/runtime identity mismatch')
        states = {name: run_state_from_mapping(value) for name, value in
                  (checkpoint or {}).get('states', {}).items()}
        methods, runs, slots = config.search['methods'], config.search['runs'], config.search['slots_per_run']
        expected = {f'{method}:{run}' for method in methods for run in range(runs)}
        if set(states) - expected:
            raise ValueError('checkpoint contains an unexpected method/run')

        def complete():
            return set(states) == expected and all(state.slots_completed == slots and
                len(state.proposals) == slots and not any(p.candidate_status == 'BILLING_PENDING' or
                (state.mode == 'LIVE' and p.request_status != 'COMPLETED') for p in state.proposals)
                for state in states.values())

        if complete():
            ledger = BudgetLedger(root/'budget', cap=runtime['cap_usd'])
            if ledger.unknown_billing or ledger.reservations or ledger.over_cap:
                raise ValueError('completed discovery has unresolved or over-cap billing')
            return _summary(root, states, ledger, 'DISCOVERY_COMPLETE')

        api_key = load_api_key(env_path)
        key_identity = hashlib.sha256(api_key.encode('utf-8')).hexdigest()
        if runtime and runtime['key_identity'] != key_identity:
            raise ValueError('API key identity changed; do not reuse the previous budget ledger')
        client = (client_factory or OpenRouterClient)(api_key=api_key)
        preferred = runtime['config']['model']['provider_slug_verified'] if runtime else None
        key, model, endpoint, remaining = _catalog(client, config, preferred)
        observed = {'model': model, 'endpoint': endpoint}
        if runtime:
            if observed != load_json_lossless(root/'catalog.json')['catalog']:
                raise ValueError('model/provider catalog or pricing changed since discovery started')
            resolved = ExperimentConfig.from_mapping(runtime['config'])
        else:
            allowance = min(remaining, _decimal(config.budget['initial_verified_allowance_usd'], 'allowance'))
            cap = validate_live_budget(True, allowance, _decimal(config.budget['local_cap_usd'], 'local cap'))
            resolved = replace(config, model={**config.model,
                'canonical_slug_verified': model['canonical_slug'],
                'provider_slug_verified': endpoint['tag'],
                'capabilities_snapshot': str(root/'catalog.json')},
                budget={**config.budget, 'initial_verified_allowance_usd': str(allowance)})
            catalog = {'artifact_class': 'LIVE_RUNTIME', 'catalog': observed, 'key': key,
                       'model': model, 'endpoint': endpoint}
            runtime = {'artifact_class': 'LIVE', 'identity': identity, 'key_identity': key_identity,
                       'config': resolved.to_mapping(), 'cap_usd': cap, 'catalog_hash': _digest(catalog),
                       'protocol_deviations': ['PILOT_SKIPPED_BY_USER'],
                       'protocol_status': 'DIRECT_DISCOVERY_WITHOUT_PILOT'}
            dump_json_atomic(root/'dataset.json', dataset)
            dump_json_atomic(root/'catalog.json', catalog)
            dump_json_atomic(runtime_path, runtime)

        ledger = BudgetLedger(root/'budget', cap=runtime['cap_usd'])

        def save():
            dump_json_atomic(checkpoint_path, {'artifact_class': 'LIVE', 'input_hash': _digest(runtime),
                'states': {name: run_state_to_mapping(state) for name, state in sorted(states.items())}})

        def finish(status, reason=None):
            ledger.refresh()
            if ledger.over_cap:
                status, reason = 'STOPPED', 'actual billed cost exceeded the local cap; no further POST allowed'
            elif status == 'DISCOVERY_COMPLETE' and (ledger.unknown_billing or ledger.reservations):
                status, reason = 'STOPPED', 'unresolved billing prevents discovery completion'
            summary = _summary(root, states, ledger, status, reason)
            dump_json_atomic(root/'summary.json', summary)
            return summary

        if ledger.unknown_billing or ledger.over_cap:
            return finish('STOPPED', 'unresolved billing or exceeded budget; inspect durable request state')
        # An existing reservation is handled by the original branch lifecycle below:
        # RESERVED may POST once; a persisted response may reconcile without POST;
        # POST_STARTED with no durable response refuses to resend.
        attempts_this_call = 0
        for slot in range(1, slots+1):
            for run in range(runs):
                ordered = [method for method in methods if method != 'RS_PARAM']
                random.Random(derive_seed(config.seed_namespace, 'round_robin', run, f'slot-{slot}')).shuffle(ordered)
                if 'RS_PARAM' in methods:
                    ordered.append('RS_PARAM')
                for method in ordered:
                    name = f'{method}:{run}'
                    state = states.get(name)
                    pending = state and state.proposals and state.proposals[-1].candidate_status == 'BILLING_PENDING'
                    if state and state.mode == 'LIVE' and state.proposals and not pending:
                        if state.proposals[-1].request_status != 'COMPLETED':
                            return finish('STOPPED', 'previous provider failure requires review')
                    if state and state.slots_completed >= slot and not pending:
                        continue
                    if pending and slot != state.slots_completed:
                        continue
                    if max_attempts is not None and attempts_this_call >= max_attempts:
                        return finish('INTERRUPTED')
                    provider = None
                    if method != 'RS_PARAM':
                        branch = root/'requests'/method/f'run-{run:03d}'
                        lifecycle = RequestLifecycle(branch, ledger,
                            context_hash=_digest({'runtime': runtime, 'method': method, 'run': run}))
                        provider = _ReservedProvider(client, lifecycle=lifecycle, estimate=Decimal('0.00000001'),
                            artifact_directory=branch/'responses', start_slot=slot)
                        provider.endpoint = endpoint
                        provider.model_context = model.get('context_length')
                        provider.catalog = observed
                    def persist(updated):
                        states[name] = updated
                        save()
                    try:
                        states[name] = run_search(method, run, resolved, provider, train['train'],
                            resume_state=state, max_new_slots=1, checkpoint_callback=persist)
                    except SearchStoppedError as error:
                        states[name] = error.state
                        save()
                        return finish('STOPPED', str(error))
                    save()
                    if ledger.over_cap:
                        return finish('STOPPED')
                    attempts_this_call += 1
                    if progress:
                        progress({'method': method, 'run_id': run, 'slot': states[name].slots_completed,
                            'target_slots': slots, 'candidate_status': states[name].proposals[-1].candidate_status,
                            'billed_usd': str(ledger.spent)})
        return finish('DISCOVERY_COMPLETE' if complete() else 'STOPPED')


def _summary(root, states, ledger, status, reason=None):
    return {'artifact_class': 'LIVE', 'status': status, 'output': str(root),
        'attempts': sum(len(state.proposals) for state in states.values()),
        'billed_usd': ledger.spent, 'cap_usd': ledger.cap,
        'completed_runs': sum(state.slots_completed == state.target_slots for state in states.values()),
        'reason': reason, 'protocol_deviations': ['PILOT_SKIPPED_BY_USER'],
        'selection_status': 'NOT_RUN', 'test_status': 'NOT_RUN'}
