"""Auditable OpenRouter REST adapter with no automatic POST retry."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .budget import RequestLifecycle, PendingRequestError, validate_live_budget
from .contracts import ModelConfig, ProviderResponse, ExperimentConfig
from .storage import dump_json_atomic, load_json_lossless


BASE_URL = "https://openrouter.ai/api/v1"
CONNECT_TIMEOUT_S = 10.0
READ_TIMEOUT_S = 120.0


class CompatibilityError(RuntimeError):
    """The frozen request cannot be supported by the verified model endpoint."""


class ProviderSchemaError(RuntimeError):
    """A provider response did not match the required public response shape."""


class HttpResponse(Protocol):
    status_code: int
    text: str

    def json(self) -> Any: ...


class HttpTransport(Protocol):
    def get(self, url: str, **kwargs: Any) -> HttpResponse: ...

    def post(self, url: str, **kwargs: Any) -> HttpResponse: ...


class RequestsTransport:
    """Thin requests transport whose adapters explicitly perform zero retries."""

    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()
        adapter = HTTPAdapter(max_retries=Retry(total=0, connect=0, read=0, status=0))
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        return self.session.get(url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> requests.Response:
        return self.session.post(url, **kwargs)


def _decimal_or_none(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return converted if converted.is_finite() and converted >= 0 else None


def _safe_error_type(error: BaseException) -> str:
    return type(error).__name__


def _as_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProviderSchemaError(f"{label} must be an object")
    return value


def _required_parameters(config: ModelConfig) -> tuple[str, ...]:
    required = ["temperature", "max_tokens", "response_format"]
    if config.reasoning_effort:
        required.append("reasoning")
    return tuple(required)


def _response_mapping(response: ProviderResponse, *, http_status: int | None) -> dict[str, Any]:
    return {
        "artifact_class": response.artifact_class,
        "http_status": http_status,
        "response_id": response.response_id,
        "content": response.content,
        "request_status": response.request_status,
        "cost_status": response.cost_status,
        "billed_cost": response.billed_cost,
        "usage": dict(response.usage),
        "finish_reason": response.finish_reason,
        "actual_model": response.actual_model,
        "actual_provider": response.actual_provider,
    }


class OpenRouterClient:
    """OpenRouter client whose constructor and imports never access the network."""

    mode = "LIVE"

    def __init__(
        self,
        *,
        api_key: str,
        transport: HttpTransport | None = None,
        base_url: str = BASE_URL,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("a non-empty OpenRouter API key is required")
        self._api_key = api_key
        self.transport = transport or RequestsTransport()
        self.base_url = base_url.rstrip("/")
        self.last_http_status: int | None = None

    def _headers(self, *, metadata: bool = False) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if metadata:
            headers["X-OpenRouter-Metadata"] = "enabled"
        return headers

    def _get_json(self, path: str) -> Mapping[str, Any]:
        response = self.transport.get(
            self.base_url + path,
            headers=self._headers(),
            timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
        )
        if response.status_code != 200:
            raise CompatibilityError(f"preflight GET failed with HTTP {response.status_code}")
        try:
            payload = response.json()
        except (ValueError, TypeError) as error:
            raise ProviderSchemaError("preflight response was not JSON") from error
        return _as_mapping(payload, label="preflight response")

    @staticmethod
    def _validate_config(config: ModelConfig) -> None:
        if not isinstance(config.model_id, str) or not config.model_id.strip():
            raise CompatibilityError("model id is required")
        if not isinstance(config.canonical_slug_verified, str) or not config.canonical_slug_verified.strip():
            raise CompatibilityError("verified canonical model slug is required before POST")
        if not isinstance(config.provider_slug_verified, str) or not config.provider_slug_verified.strip():
            raise CompatibilityError("verified provider slug is required before POST")
        if not isinstance(config.temperature, (int, float)) or isinstance(config.temperature, bool):
            raise CompatibilityError("temperature must be numeric")
        if type(config.max_tokens) is not int or config.max_tokens <= 0:
            raise CompatibilityError("max_tokens must be a positive integer")
        if config.response_format != "json_object":
            raise CompatibilityError("response_format must be json_object")
        if config.stream is not False:
            raise CompatibilityError("streaming is not permitted by this protocol")
        if config.allow_model_fallbacks:
            raise CompatibilityError("model fallbacks are not permitted")
        if config.allow_provider_fallbacks:
            raise CompatibilityError("provider fallbacks are not permitted")
        if config.require_parameters is not True:
            raise CompatibilityError("provider parameter enforcement is required")

    def get_key_info(self) -> dict[str, Any]:
        payload = self._get_json("/key")
        data = _as_mapping(payload.get("data"), label="key data")
        result: dict[str, Any] = {}
        for name in (
            "limit",
            "limit_remaining",
            "usage",
            "usage_daily",
            "usage_weekly",
            "usage_monthly",
            "byok_usage",
            "byok_usage_daily",
            "byok_usage_weekly",
            "byok_usage_monthly",
        ):
            if name in data:
                numeric = _decimal_or_none(data[name])
                if numeric is None:
                    raise ProviderSchemaError(f"key field {name} is not a finite non-negative number")
                result[name] = numeric
        for name in ("is_free_tier", "include_byok_in_limit", "expires_at", "limit_reset"):
            if name in data:
                result[name] = data[name]
        return result

    def get_model(self, model_id: str) -> dict[str, Any]:
        payload = self._get_json(f"/model/{model_id}")
        data = _as_mapping(payload.get("data"), label="model data")
        return {
            "id": data.get("id"),
            "canonical_slug": data.get("canonical_slug"),
            "supported_parameters": list(data.get("supported_parameters") or []),
            "pricing": dict(data.get("pricing") or {}),
            "architecture": dict(data.get("architecture") or {}),
            "context_length": data.get("context_length"),
            "top_provider": dict(data.get("top_provider") or {}),
        }

    def get_endpoints(self, model_id: str) -> list[dict[str, Any]]:
        payload = self._get_json(f"/models/{model_id}/endpoints")
        data = _as_mapping(payload.get("data"), label="endpoint catalog data")
        endpoints = data.get("endpoints")
        if not isinstance(endpoints, list):
            raise ProviderSchemaError("endpoint catalog must contain an endpoints list")
        sanitized: list[dict[str, Any]] = []
        for endpoint in endpoints:
            item = _as_mapping(endpoint, label="endpoint")
            sanitized.append(
                {
                    "tag": item.get("tag"),
                    "provider_name": item.get("provider_name"),
                    "status": item.get("status"),
                    "max_completion_tokens": item.get("max_completion_tokens"),
                    "supported_parameters": list(item.get("supported_parameters") or []),
                    "pricing": dict(item.get("pricing") or {}),
                }
            )
        return sanitized

    def preflight(
        self,
        config: ModelConfig,
        *,
        artifact_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Fetch and validate a current key/model/endpoint snapshot."""
        self._validate_config(config)
        key = self.get_key_info()
        model = self.get_model(config.model_id)
        endpoints = self.get_endpoints(config.model_id)
        errors: list[str] = []
        if model["id"] != config.model_id:
            errors.append("catalog model id does not match requested model id")
        if model["canonical_slug"] != config.canonical_slug_verified:
            errors.append("catalog canonical slug does not match verified canonical slug")
        required = set(_required_parameters(config))
        model_missing = sorted(required - set(model["supported_parameters"]))
        if model_missing:
            errors.append("model is missing parameters: " + ", ".join(model_missing))
        selected = next(
            (
                endpoint
                for endpoint in endpoints
                if endpoint.get("tag") == config.provider_slug_verified
            ),
            None,
        )
        if selected is None:
            errors.append("verified provider is absent from endpoint catalog")
        else:
            endpoint_missing = sorted(
                required - set(selected.get("supported_parameters") or [])
            )
            if endpoint_missing:
                errors.append(
                    "provider is missing parameters: " + ", ".join(endpoint_missing)
                )
            max_completion = selected.get("max_completion_tokens")
            if (
                isinstance(max_completion, int)
                and max_completion < config.max_tokens
            ):
                errors.append("provider max completion tokens are below configured max_tokens")
        snapshot = {
            "artifact_class": "LIVE_PREFLIGHT",
            "key": key,
            "model": model,
            "endpoint": selected,
            "compatibility": {
                "compatible": not errors,
                "required_parameters": tuple(sorted(required)),
                "errors": tuple(errors),
            },
        }
        if artifact_path is not None:
            dump_json_atomic(artifact_path, snapshot)
        if errors:
            raise CompatibilityError("; ".join(errors))
        return snapshot

    def _request_payload(
        self,
        messages: Sequence[Mapping[str, str]],
        config: ModelConfig,
    ) -> dict[str, Any]:
        self._validate_config(config)
        if not messages:
            raise ValueError("at least one message is required")
        normalized_messages: list[dict[str, str]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise ValueError("each message requires string role and content")
            normalized_messages.append({"role": role, "content": content})
        return {
            "model": config.model_id,
            "messages": normalized_messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "response_format": {"type": config.response_format},
            "reasoning": {"effort": config.reasoning_effort},
            "provider": {
                "only": [config.provider_slug_verified],
                "allow_fallbacks": config.allow_provider_fallbacks,
                "require_parameters": config.require_parameters,
            },
            "stream": config.stream,
        }

    def propose(
        self,
        messages: list[dict[str, str]],
        config: ModelConfig,
    ) -> ProviderResponse:
        """Issue exactly one POST and return billing independently of candidate parsing."""
        payload = self._request_payload(messages, config)
        self.last_http_status = None
        try:
            http_response = self.transport.post(
                self.base_url + "/chat/completions",
                headers=self._headers(metadata=True),
                json=payload,
                timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
            )
        except (TimeoutError, requests.Timeout) as error:
            return ProviderResponse(
                response_id=None,
                content=None,
                request_status="TIMEOUT",
                cost_status="UNKNOWN_BILLING",
                billed_cost=None,
                usage={"error_type": _safe_error_type(error)},
                artifact_class="LIVE",
            )
        except requests.RequestException as error:
            return ProviderResponse(
                response_id=None,
                content=None,
                request_status="TRANSPORT_ERROR",
                cost_status="UNKNOWN_BILLING",
                billed_cost=None,
                usage={"error_type": _safe_error_type(error)},
                artifact_class="LIVE",
            )
        self.last_http_status = http_response.status_code
        if http_response.status_code != 200:
            return ProviderResponse(
                response_id=None,
                content=None,
                request_status=f"HTTP_{http_response.status_code}",
                cost_status="UNKNOWN_BILLING",
                billed_cost=None,
                usage={"http_status": http_response.status_code},
                artifact_class="LIVE",
            )
        try:
            raw = _as_mapping(http_response.json(), label="completion response")
        except (ValueError, TypeError, ProviderSchemaError) as error:
            return ProviderResponse(
                response_id=None,
                content=None,
                request_status="PROVIDER_SCHEMA_ERROR",
                cost_status="UNKNOWN_BILLING",
                billed_cost=None,
                usage={"error_type": _safe_error_type(error)},
                artifact_class="LIVE",
            )
        usage_value = raw.get("usage")
        usage = dict(usage_value) if isinstance(usage_value, Mapping) else {}
        billed_cost = _decimal_or_none(usage.get("cost"))
        cost_status = "BILLED" if billed_cost is not None else "UNKNOWN_BILLING"
        choices = raw.get("choices")
        content: str | None = None
        finish_reason: str | None = None
        schema_valid = isinstance(choices, list) and bool(choices)
        if schema_valid:
            choice = choices[0]
            schema_valid = isinstance(choice, Mapping)
        if schema_valid:
            message = choice.get("message")  # type: ignore[union-attr]
            schema_valid = isinstance(message, Mapping)
        if schema_valid:
            candidate_content = message.get("content")  # type: ignore[union-attr]
            schema_valid = candidate_content is None or isinstance(candidate_content, str)
            content = candidate_content if schema_valid else None
            reason = choice.get("finish_reason")  # type: ignore[union-attr]
            finish_reason = reason if isinstance(reason, str) else None
        actual_model = raw.get("model") if isinstance(raw.get("model"), str) else None
        actual_provider = raw.get("provider") if isinstance(raw.get("provider"), str) else None
        if actual_provider is None:
            metadata = raw.get("openrouter_metadata")
            if isinstance(metadata, Mapping):
                endpoints = metadata.get("endpoints")
                available = endpoints.get("available") if isinstance(endpoints, Mapping) else None
                if isinstance(available, list):
                    selected = next(
                        (
                            item
                            for item in available
                            if isinstance(item, Mapping) and item.get("selected") is True
                        ),
                        None,
                    )
                    if selected is not None and isinstance(selected.get("provider"), str):
                        actual_provider = selected["provider"]
        request_status = "COMPLETED"
        if not schema_valid:
            request_status = "PROVIDER_SCHEMA_ERROR"
        elif actual_model not in {config.model_id, config.canonical_slug_verified}:
            request_status = "MODEL_MISMATCH"
        else:
            allowed_providers = {config.provider_slug_verified}
            if config.capabilities_snapshot:
                snapshot = load_json_lossless(config.capabilities_snapshot)
                endpoint = snapshot.get('endpoint') or {}
                if endpoint.get('tag') == config.provider_slug_verified:
                    allowed_providers.add(endpoint.get('provider_name'))
            if actual_provider is None or actual_provider not in allowed_providers:
                request_status = 'PROVIDER_MISMATCH'
        response_id = raw.get("id") if isinstance(raw.get("id"), str) else None
        return ProviderResponse(
            response_id=response_id,
            content=content,
            request_status=request_status,
            cost_status=cost_status,
            billed_cost=billed_cost,
            usage=usage,
            finish_reason=finish_reason,
            actual_model=actual_model,
            actual_provider=actual_provider,
            artifact_class="LIVE",
        )

    def propose_managed(
        self,
        messages: list[dict[str, str]],
        config: ModelConfig,
        *,
        lifecycle: RequestLifecycle,
        slot: int,
        estimate: Decimal,
        artifact_path: str | Path,
    ) -> ProviderResponse:
        """Persist the provider response and reconcile billing before parsing content."""
        payload = self._request_payload(messages, config)
        request_hash = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        artifact = Path(artifact_path)
        if lifecycle.state is not None and lifecycle.state.get('slot') == slot and artifact.exists():
            saved = load_json_lossless(artifact)
            request_id = lifecycle.state['request_id']
            if (saved.get('request_id') != request_id
                or saved.get('context_hash') != lifecycle.context_hash
                or saved.get('request_hash') != request_hash):
                raise PendingRequestError('saved response request/context hash mismatch')
            response = ProviderResponse(**{key:saved[key] for key in ProviderResponse.__dataclass_fields__})
            if lifecycle.phase == 'POST_STARTED':
                lifecycle.record_response(request_id,response_id=response.response_id,actual_cost=response.billed_cost)
            if lifecycle.phase == 'RESPONSE_RECEIVED':
                lifecycle.reconcile_response(request_id)
            if lifecycle.phase == 'RECONCILED':
                lifecycle.checkpoint(request_id)
            if lifecycle.phase == 'UNKNOWN_BILLING':
                raise PendingRequestError('saved response billing must be resolved before search recovery')
            if lifecycle.phase != 'CHECKPOINTED':
                raise PendingRequestError('saved response is not in a recoverable phase')
            if response.billed_cost is None:
                resolved = next((event['actual_cost'] for event in reversed(lifecycle.ledger.events)
                    if event.get('request_id') == request_id and event['event'] == 'BILLING_RESOLVED'), None)
                if resolved is None:
                    raise PendingRequestError('checkpointed response has no verified billing evidence')
                response = replace(response, billed_cost=resolved, cost_status='BILLED')
            return response
        request_id = lifecycle.start_slot(slot, estimate)
        lifecycle.mark_post_started(request_id)
        response = self.propose(messages, config)
        dump_json_atomic(
            artifact_path,
            {
                "request_id": request_id,
                "slot": slot,
                "context_hash": lifecycle.context_hash,
                "request_hash": request_hash,
                **_response_mapping(response, http_status=self.last_http_status),
            },
        )
        lifecycle.record_response(
            request_id,
            response_id=response.response_id,
            actual_cost=response.billed_cost,
        )
        lifecycle.reconcile_response(request_id)
        if response.billed_cost is not None:
            lifecycle.checkpoint(request_id)
        return response


class ManagedOpenRouterProvider:
    """Search adapter requiring explicit live config and a durable budget lifecycle."""

    mode = 'LIVE'

    def __init__(self, client: OpenRouterClient, *, lifecycle: RequestLifecycle,
                 estimate: Decimal, artifact_directory: str | Path, start_slot: int = 1):
        self.client = client
        self.lifecycle = lifecycle
        self.estimate = estimate
        self.artifact_directory = Path(artifact_directory)
        self.slot = start_slot

    def propose(self, messages: list[dict[str,str]], config: ExperimentConfig) -> ProviderResponse:
        if config.mode != 'LIVE' or config.allow_live is not True:
            raise CompatibilityError('managed LIVE provider requires explicit approval')
        try:
            cap = validate_live_budget(config.allow_live,
                Decimal(str(config.budget.get('initial_verified_allowance_usd'))),
                Decimal(str(config.budget.get('local_cap_usd'))))
        except (ValueError, ArithmeticError, RuntimeError) as error:
            raise CompatibilityError('managed provider requires verified finite budget') from error
        if self.lifecycle.ledger.cap > cap:
            raise CompatibilityError('ledger cap exceeds approved planning cap')
        response = self.client.propose_managed(messages, ModelConfig.from_mapping(config.model),
            lifecycle=self.lifecycle,slot=self.slot,estimate=self.estimate,
            artifact_path=self.artifact_directory/f'slot-{self.slot:04d}.json')
        self.slot += 1
        return response


def snapshot_hash(snapshot: Mapping[str, Any]) -> str:
    """Hash a snapshot using the same lossless values written to artifacts."""
    payload = json.dumps(snapshot, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
