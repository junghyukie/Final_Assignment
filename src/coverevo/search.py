"""Common deterministic search controller for IND, EVO-P, EVO-F and RS-PARAM."""

from __future__ import annotations

from dataclasses import asdict, replace
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import math
import random
from typing import Any, Callable, Mapping, Protocol, Sequence

from .contracts import (
    ArchiveEntry,
    EvalResult,
    ExperimentConfig,
    Instance,
    Program,
    ProposalRecord,
    ProviderResponse,
    RunState,
)
from .dsl import DSLValidationError, parse_program
from .evaluator import training_fitness
from .instances import derive_seed
from .prompts import CandidateResponseError, build_messages, parse_candidate_response
from .solver import greedy_reference


class ProposalProvider(Protocol):
    mode: str

    def propose(
        self, messages: list[dict[str, str]], config: ExperimentConfig
    ) -> ProviderResponse: ...


class SearchStoppedError(RuntimeError):
    def __init__(self, state: RunState, reason: str):
        super().__init__(reason)
        self.state = state


class FixtureProposalProvider:
    """Replay explicit scripted response content without network access."""

    mode = "MOCK"

    def __init__(self, responses: Sequence[str | ProviderResponse | Exception]) -> None:
        self._responses = list(responses)
        self._position = 0
        self.calls: list[list[dict[str, str]]] = []

    def propose(
        self, messages: list[dict[str, str]], config: ExperimentConfig
    ) -> ProviderResponse:
        del config
        self.calls.append([dict(message) for message in messages])
        if self._position >= len(self._responses):
            raise RuntimeError("fixture provider has no scripted response for this slot")
        scripted = self._responses[self._position]
        self._position += 1
        if isinstance(scripted, Exception):
            raise scripted
        if isinstance(scripted, ProviderResponse):
            return scripted
        return ProviderResponse(
            response_id=f"mock-response-{self._position:04d}",
            content=scripted,
            request_status="MOCK_RESPONSE",
            cost_status="NOT_APPLICABLE_MOCK",
            billed_cost=None,
            usage={},
            finish_reason="stop",
            artifact_class="MOCK",
        )


def _required_search_int(config: ExperimentConfig, name: str) -> int:
    value = config.search.get(name)
    if type(value) is not int or value <= 0:
        raise ValueError(f"search.{name} must be a positive integer")
    return value


def is_fresh_slot(method: str, slot: int, config: ExperimentConfig) -> bool:
    """Return the locked 1-based proposal schedule decision."""
    normalized = method.upper()
    if type(slot) is not int or slot < 1:
        raise ValueError("slot must be a positive 1-based integer")
    if normalized == "IND":
        return True
    if normalized not in {"EVO_P", "EVO_F"}:
        return False
    initial = _required_search_int(config, "initial_fresh_slots")
    period = _required_search_int(config, "mutation_fresh_period")
    return slot <= initial or (slot >= initial + 1 and (slot - (initial + 1)) % period == period - 1)


def _prompt_hash(messages: list[dict[str, str]]) -> str:
    encoded = json.dumps(
        messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _behavior_fingerprint(evaluation: EvalResult) -> str:
    rows = [
        {
            "instance_id": row["instance_id"],
            "success": row["success"],
            "selected": list(row["selected"]),
            "raw_cost": row["raw_cost"],
        }
        for row in evaluation.per_instance
    ]
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ranking_key(entry: ArchiveEntry) -> tuple[Fraction, int, str]:
    if entry.evaluation.mean_ratio is None:
        raise ValueError("invalid evaluation cannot be ranked in archive")
    return (
        entry.evaluation.mean_ratio,
        entry.program.ast_nodes,
        entry.program.program_hash,
    )


def _insert_archive(
    archive: list[ArchiveEntry], entry: ArchiveEntry, archive_size: int
) -> None:
    archive.append(entry)
    archive.sort(key=_ranking_key)
    del archive[archive_size:]


def _rankdata(values: list[int]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def _spearman(instance: Instance) -> float | None:
    degrees = [0] * instance.n
    for u, v in instance.edges:
        degrees[u] += 1
        degrees[v] += 1
    degree_ranks = _rankdata(degrees)
    cost_ranks = _rankdata(list(instance.costs))
    degree_mean = sum(degree_ranks) / instance.n
    cost_mean = sum(cost_ranks) / instance.n
    left = [value - degree_mean for value in degree_ranks]
    right = [value - cost_mean for value in cost_ranks]
    denominator = math.sqrt(
        sum(value * value for value in left) * sum(value * value for value in right)
    )
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(left, right)) / denominator


def _feedback_for_entry(
    entry: ArchiveEntry, train_by_id: dict[str, Instance]
) -> dict[str, Any]:
    if entry.evaluation.mean_ratio is None:
        raise ValueError("feedback requires a valid training evaluation")
    ratios = sorted(row["ratio"] for row in entry.evaluation.per_instance)
    nearest_rank = max(0, math.ceil(0.9 * len(ratios)) - 1)
    wins = sum(ratio < 1 for ratio in ratios)
    ties = sum(ratio == 1 for ratio in ratios)
    losses = sum(ratio > 1 for ratio in ratios)
    worst_rows = sorted(
        entry.evaluation.per_instance,
        key=lambda row: (row["ratio"], row["instance_id"]),
        reverse=True,
    )[:2]
    cases: list[dict[str, Any]] = []
    for row in worst_rows:
        instance = train_by_id[row["instance_id"]]
        degrees = [0] * instance.n
        for u, v in instance.edges:
            degrees[u] += 1
            degrees[v] += 1
        cases.append(
            {
                "instance_id": instance.instance_id,
                "n": instance.n,
                "mean_degree": Fraction(2 * len(instance.edges), instance.n),
                "isolated_nodes": sum(degree == 0 for degree in degrees),
                "cost_degree_spearman": _spearman(instance),
                "ratio": row["ratio"],
            }
        )
    return {
        "program_hash": entry.program.program_hash,
        "mean_ratio": entry.evaluation.mean_ratio,
        "p90_ratio": ratios[nearest_rank],
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "worst_cases": cases,
    }


def _coefficient_literal(value: int) -> str:
    decimal = f"{abs(value) / 1000:.3f}".rstrip("0").rstrip(".")
    if not decimal:
        decimal = "0"
    return f"(-{decimal})" if value < 0 else decimal


def _parameter_response(rng: random.Random, slot: int) -> ProviderResponse:
    a, b, d = (rng.randint(-2000, 2000) for _ in range(3))
    expression = (
        "div(gain,cost)*(1+"
        f"{_coefficient_literal(a)}*rarity+"
        f"{_coefficient_literal(b)}*overlap+"
        f"{_coefficient_literal(d)}*rarity*remaining)"
    )
    content = json.dumps(
        {
            "expression": expression,
            "rationale": f"RS-PARAM fixture coefficients a={a}, b={b}, d={d}.",
        },
        separators=(",", ":"),
    )
    return ProviderResponse(
        response_id=f"parameter-{slot:04d}",
        content=content,
        request_status="PARAMETER_PROPOSAL",
        cost_status="NOT_APPLICABLE_OFFLINE",
        billed_cost=None,
        usage={},
        artifact_class="OFFLINE",
    )


def run_search(
    method: str,
    run_id: int,
    config: ExperimentConfig,
    provider: ProposalProvider | None,
    train: list[Instance],
    *,
    resume_state: RunState | None = None,
    max_new_slots: int | None = None,
    checkpoint_callback: Callable[[RunState], None] | None = None,
) -> RunState:
    """Run one deterministic proposal budget through a common evaluation path."""
    normalized = method.upper()
    if normalized not in {"IND", "EVO_P", "EVO_F", "RS_PARAM"}:
        raise ValueError(f"unknown search method: {method}")
    if type(run_id) is not int or run_id < 0:
        raise ValueError("run_id must be a non-negative integer")
    if not train or any(instance.split != "train" for instance in train):
        raise ValueError("search may access training instances only")
    if normalized != "RS_PARAM" and provider is None:
        raise ValueError("LLM search methods require a proposal provider")
    if normalized != "RS_PARAM" and provider is not None and provider.mode != "MOCK":
        from .openrouter_client import ManagedOpenRouterProvider
        if (not isinstance(provider, ManagedOpenRouterProvider)
            or config.allow_live is not True or config.mode != 'LIVE'):
            raise ValueError('LIVE search requires approved config and managed provider')

    slots = _required_search_int(config, "slots_per_run")
    archive_size = _required_search_int(config, "archive_size")
    reference_costs = {
        instance.instance_id: greedy_reference(instance).raw_cost for instance in train
    }
    if any(cost is None for cost in reference_costs.values()):
        raise RuntimeError("greedy reference failed on training data")
    exact_reference_costs = {
        identifier: int(cost) for identifier, cost in reference_costs.items()
    }

    experiment_material = f"{config.protocol_version}|{normalized}|{run_id}"
    experiment_id = hashlib.sha256(experiment_material.encode("utf-8")).hexdigest()[:16]
    rng = random.Random(
        derive_seed(config.seed_namespace, normalized.lower(), run_id, "search")
    )
    if resume_state is None:
        seed_program = replace(
            parse_program("div(gain, cost)", "handwritten_seed"),
            rationale="Registered greedy seed.",
        )
        seed_evaluation = training_fitness(seed_program, train, exact_reference_costs)
        if not seed_evaluation.valid_on_all_instances:
            raise RuntimeError("registered seed failed on training data")
        seed_entry = ArchiveEntry(
            program=seed_program,
            evaluation=seed_evaluation,
            behavior_hash=_behavior_fingerprint(seed_evaluation),
            source_slot=0,
        )
        state = RunState(
            experiment_id=experiment_id,
            method=normalized,
            run_id=run_id,
            mode="OFFLINE" if normalized == "RS_PARAM" else provider.mode,
            slots_completed=0,
            archive=[seed_entry],
            programs={seed_program.program_hash: seed_program},
            evaluations={seed_program.program_hash: seed_evaluation},
            proposals=[],
            raw_responses=[],
            prompt_records=[],
            syntactic_unique_count=1,
            behavior_unique_count=1,
            rng_state=rng.getstate(),
            target_slots=slots,
        )
    else:
        state = resume_state
        if (
            state.experiment_id != experiment_id
            or state.method != normalized
            or state.run_id != run_id
        ):
            raise ValueError("resume state does not match method/run/config identity")
        expected_mode = "OFFLINE" if normalized == "RS_PARAM" else provider.mode
        if state.mode != expected_mode:
            raise ValueError("resume state mode mismatch")
        if not 0 <= state.slots_completed <= slots:
            raise ValueError("resume state slot is outside configured budget")
        if state.target_slots != slots:
            raise ValueError("resume state target slot budget mismatch")
        if state.rng_state is None:
            raise ValueError("resume state is missing RNG state")
        rng.setstate(state.rng_state)
        if state.proposals and state.proposals[-1].candidate_status == 'BILLING_PENDING':
            from .openrouter_client import ManagedOpenRouterProvider
            if not isinstance(provider, ManagedOpenRouterProvider):
                raise SearchStoppedError(state,'pending billing requires the original managed provider')
            provider.lifecycle.ledger.refresh()
            provider.lifecycle._synchronize_from_log()
            if provider.lifecycle.phase not in {'RECONCILED','CHECKPOINTED'}:
                raise SearchStoppedError(state,'billing must be resolved before candidate recovery')
            if state.pending_rng_state is None:
                raise SearchStoppedError(state,'pending slot is missing its pre-request RNG state')
            provider.slot = state.proposals[-1].slot
            rng.setstate(state.pending_rng_state)
            state.proposals.pop()
            state.raw_responses.pop()
            state.prompt_records.pop()
            state.slots_completed -= 1
            state.pending_rng_state = None
    if max_new_slots is not None and (
        type(max_new_slots) is not int or max_new_slots < 0
    ):
        raise ValueError("max_new_slots must be a non-negative integer or None")
    train_by_id = {instance.instance_id: instance for instance in train}

    first_slot = state.slots_completed + 1
    last_slot = slots
    if max_new_slots is not None:
        last_slot = min(last_slot, state.slots_completed + max_new_slots)
    for slot in range(first_slot, last_slot + 1):
        slot_rng_state = rng.getstate()
        parents: list[ArchiveEntry] = []
        messages: list[dict[str, str]] = []
        if normalized == "RS_PARAM":
            response = _parameter_response(rng, slot)
            prompt_hash = hashlib.sha256(b"").hexdigest()
        else:
            fresh = is_fresh_slot(normalized, slot, config)
            if not fresh:
                sample_size = min(2, len(state.archive))
                parents = rng.sample(state.archive, sample_size)
                parents.sort(key=lambda entry: entry.program.program_hash)
            if fresh:
                messages = build_messages("IND", [], None, config)
            elif normalized == "EVO_P":
                messages = build_messages(
                    "EVO_P", [entry.program for entry in parents], None, config
                )
            else:
                feedback = [
                    _feedback_for_entry(entry, train_by_id) for entry in parents
                ]
                messages = build_messages(
                    "EVO_F", [entry.program for entry in parents], feedback, config
                )
            prompt_hash = _prompt_hash(messages)
            state.prompt_records.append(
                {
                    "slot": slot,
                    "kind": "FRESH" if fresh else "PARENT",
                    "prompt_hash": prompt_hash,
                    "messages": messages,
                }
            )
            try:
                response = provider.propose(messages, config)  # type: ignore[union-attr]
            except Exception as error:
                if provider.mode == 'LIVE':
                    # No candidate was committed. Retry the same slot with its original
                    # RNG and let the durable provider recover, never duplicate its prompt.
                    state.prompt_records.pop()
                    state.rng_state = slot_rng_state
                    raise SearchStoppedError(state,'managed provider failed; inspect durable request state') from error
                response = ProviderResponse(
                    response_id=None,
                    content=None,
                    request_status="PROVIDER_ERROR",
                    cost_status="NOT_APPLICABLE_MOCK",
                    billed_cost=None,
                    usage={"error_type": type(error).__name__},
                    artifact_class="MOCK",
                )

        state.raw_responses.append(
            {
                "slot": slot,
                "artifact_class": response.artifact_class,
                "response_id": response.response_id,
                "content": response.content,
                "request_status": response.request_status,
                "cost_status": response.cost_status,
                "usage": dict(response.usage),
            }
        )
        candidate_status = "PROVIDER_ERROR"
        program_hash: str | None = None
        billing_pending = response.artifact_class == 'LIVE' and response.billed_cost is None
        if billing_pending:
            candidate_status = 'BILLING_PENDING'
            state.pending_rng_state = slot_rng_state
        if not billing_pending and response.request_status in {"MOCK_RESPONSE", "PARAMETER_PROPOSAL", "COMPLETED"}:
            try:
                expression, rationale = parse_candidate_response(response.content)
                program = replace(
                    parse_program(
                        expression,
                        "parameter_search" if normalized == "RS_PARAM" else "llm",
                    ),
                    rationale=rationale,
                )
                program_hash = program.program_hash
                if program_hash in state.programs:
                    candidate_status = "DUPLICATE"
                else:
                    evaluation = training_fitness(
                        program, train, exact_reference_costs
                    )
                    state.programs[program_hash] = program
                    state.evaluations[program_hash] = evaluation
                    if not evaluation.valid_on_all_instances:
                        reasons = {
                            failure["failure_reason"] for failure in evaluation.failures
                        }
                        candidate_status = (
                            "NUMERIC_LIMIT"
                            if reasons == {"NUMERIC_LIMIT"}
                            else "EVALUATION_ERROR"
                        )
                    else:
                        candidate_status = "VALID"
                        entry = ArchiveEntry(
                            program=program,
                            evaluation=evaluation,
                            behavior_hash=_behavior_fingerprint(evaluation),
                            source_slot=slot,
                        )
                        _insert_archive(state.archive, entry, archive_size)
            except CandidateResponseError as error:
                candidate_status = error.status
            except DSLValidationError:
                candidate_status = "INVALID_AST"

        state.proposals.append(
            ProposalRecord(
                experiment_id=state.experiment_id,
                method=normalized,
                run_id=run_id,
                slot=slot,
                prompt_hash=prompt_hash,
                response_id=response.response_id,
                program_hash=program_hash,
                request_status=response.request_status,
                candidate_status=candidate_status,
                cost_status=response.cost_status,
                billed_cost=response.billed_cost,
                usage=dict(response.usage),
                parent_hashes=tuple(
                    entry.program.program_hash for entry in parents
                ),
            )
        )
        state.slots_completed = slot
        state.rng_state = rng.getstate()
        if checkpoint_callback is not None:
            checkpoint_callback(state)
        if response.artifact_class == 'LIVE' and (
            response.request_status != 'COMPLETED' or response.billed_cost is None
        ):
            raise SearchStoppedError(state,'LIVE request requires review: '+response.request_status+'/'+response.cost_status)

    state.syntactic_unique_count = len(state.programs)
    state.behavior_unique_count = len(
        {
            _behavior_fingerprint(evaluation)
            for evaluation in state.evaluations.values()
            if evaluation.valid_on_all_instances
        }
    )
    return state


def run_state_to_mapping(state: RunState) -> dict[str, Any]:
    """Serialize resumable search state without duplicating archive semantics."""
    return {
        "experiment_id": state.experiment_id,
        "method": state.method,
        "run_id": state.run_id,
        "mode": state.mode,
        "slots_completed": state.slots_completed,
        "programs": {
            key: asdict(value) for key, value in sorted(state.programs.items())
        },
        "evaluations": {
            key: asdict(value) for key, value in sorted(state.evaluations.items())
        },
        "archive": [
            {
                "program_hash": entry.program.program_hash,
                "behavior_hash": entry.behavior_hash,
                "source_slot": entry.source_slot,
            }
            for entry in state.archive
        ],
        "proposals": [asdict(record) for record in state.proposals],
        "raw_responses": state.raw_responses,
        "prompt_records": state.prompt_records,
        "syntactic_unique_count": state.syntactic_unique_count,
        "behavior_unique_count": state.behavior_unique_count,
        "rng_state": state.rng_state,
        "target_slots": state.target_slots,
        "pending_rng_state": state.pending_rng_state,
    }


def run_state_from_mapping(data: Mapping[str, Any]) -> RunState:
    """Restore a state emitted by :func:`run_state_to_mapping`."""
    programs = {
        key: Program(**value) for key, value in dict(data["programs"]).items()
    }
    evaluations = {
        key: EvalResult(**value) for key, value in dict(data["evaluations"]).items()
    }
    archive = [
        ArchiveEntry(
            program=programs[item["program_hash"]],
            evaluation=evaluations[item["program_hash"]],
            behavior_hash=item["behavior_hash"],
            source_slot=item["source_slot"],
        )
        for item in data["archive"]
    ]
    proposals = [
        ProposalRecord(
            experiment_id=item["experiment_id"],
            method=item["method"],
            run_id=item["run_id"],
            slot=item["slot"],
            prompt_hash=item["prompt_hash"],
            response_id=item["response_id"],
            program_hash=item["program_hash"],
            request_status=item["request_status"],
            candidate_status=item["candidate_status"],
            cost_status=item["cost_status"],
            billed_cost=item["billed_cost"],
            usage=dict(item["usage"]),
            parent_hashes=tuple(item["parent_hashes"]),
        )
        for item in data["proposals"]
    ]
    return RunState(
        experiment_id=data["experiment_id"],
        method=data["method"],
        run_id=data["run_id"],
        mode=data["mode"],
        slots_completed=data["slots_completed"],
        archive=archive,
        programs=programs,
        evaluations=evaluations,
        proposals=proposals,
        raw_responses=list(data["raw_responses"]),
        prompt_records=list(data["prompt_records"]),
        syntactic_unique_count=data["syntactic_unique_count"],
        behavior_unique_count=data["behavior_unique_count"],
        rng_state=data.get("rng_state"),
        target_slots=data.get("target_slots"),
        pending_rng_state=data.get("pending_rng_state"),
    )
