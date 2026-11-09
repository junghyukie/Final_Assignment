"""Leakage-safe fitness, benchmark records and replayable evaluation cache."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .contracts import EvalResult, Instance, Program, SolveResult
from .solver import solve
from .storage import (
    dump_json_atomic,
    load_json_lossless,
    solve_result_from_mapping,
    solve_result_to_mapping,
)


class SplitAccessError(ValueError):
    """Raised when discovery fitness receives non-training data."""


@dataclass(frozen=True, slots=True)
class EvaluationVersions:
    solver_version: str = "solver-v1"
    dsl_version: str = "dsl-v1"
    feature_version: str = "features-v1"
    pruning_version: str = "pruning-v1"
    tie_break_version: str = "score-cost-id-v1"
    evaluator_version: str = "evaluator-v1"


def evaluation_cache_key(
    program: Program, instance: Instance, versions: EvaluationVersions
) -> str:
    payload = {
        "program_hash": program.program_hash,
        "instance_hash": instance.content_hash,
        "versions": asdict(versions),
        "prune": True,
        "pruning_order": "cost_desc_then_id_asc",
        "score_tie_break": "cost_asc_then_id_asc",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


class EvaluationCache:
    """Small persistent cache with deterministic hit/miss audit records."""

    SCHEMA_VERSION = "evaluation-cache-v1"

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._entries: dict[str, SolveResult] = {}
        self.audit_records: list[dict[str, Any]] = []
        if self.path is not None and self.path.exists():
            payload = load_json_lossless(self.path)
            if payload.get("schema_version") != self.SCHEMA_VERSION:
                raise ValueError("evaluation cache schema version mismatch")
            self._entries = {
                key: solve_result_from_mapping(value)
                for key, value in payload.get("entries", {}).items()
            }
            self.audit_records = list(payload.get("audit_records", []))

    def _persist(self) -> None:
        if self.path is None:
            return
        dump_json_atomic(
            self.path,
            {
                "schema_version": self.SCHEMA_VERSION,
                "entries": {
                    key: solve_result_to_mapping(value)
                    for key, value in sorted(self._entries.items())
                },
                "audit_records": self.audit_records,
            },
        )

    def get(self, key: str) -> SolveResult | None:
        result = self._entries.get(key)
        self.audit_records.append(
            {"event": "HIT" if result is not None else "MISS", "cache_key": key}
        )
        self._persist()
        return result

    def put(self, key: str, result: SolveResult) -> None:
        existing = self._entries.get(key)
        if existing is not None and existing != result:
            raise ValueError("cache key collision with different evaluation result")
        self._entries[key] = result
        self.audit_records.append({"event": "STORE", "cache_key": key})
        self._persist()


def evaluate_program(
    program: Program,
    instance: Instance,
    *,
    cache: EvaluationCache | None = None,
    versions: EvaluationVersions | None = None,
) -> SolveResult:
    active_versions = versions or EvaluationVersions()
    key = evaluation_cache_key(program, instance, active_versions)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached
    result = solve(instance, program, prune=True, trace=False)
    if cache is not None:
        cache.put(key, result)
    return result


def _validate_inputs(
    instances: list[Instance], reference_costs: dict[str, int]
) -> None:
    if not instances:
        raise ValueError("evaluation requires at least one instance")
    identifiers = [instance.instance_id for instance in instances]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("evaluation instances must have unique instance_id values")
    missing = [identifier for identifier in identifiers if identifier not in reference_costs]
    if missing:
        raise ValueError(f"missing reference costs for: {missing}")
    for identifier in identifiers:
        value = reference_costs[identifier]
        if type(value) is not int or value <= 0:
            raise ValueError(f"reference cost for {identifier} must be a positive integer")


def _evaluation_row(
    program: Program,
    instance: Instance,
    result: SolveResult,
    reference_cost: int,
) -> dict[str, Any]:
    raw_ratio = (
        Fraction(result.raw_cost, reference_cost) if result.raw_cost is not None else None
    )
    return {
        "program_hash": program.program_hash,
        "instance_id": instance.instance_id,
        "base_graph_id": instance.base_graph_id,
        "split": instance.split,
        "variant": instance.variant,
        "success": result.success,
        "selected_before_pruning": result.selected_before_pruning,
        "selected": result.selected,
        "cost_before_pruning": result.cost_before_pruning,
        "raw_cost": result.raw_cost,
        "operational_cost": result.operational_cost,
        "reference_cost": reference_cost,
        "ratio": raw_ratio,
        "operational_ratio": Fraction(result.operational_cost, reference_cost),
        "failure_reason": result.failure_reason,
        "runtime_ns": result.runtime_ns,
    }


def training_fitness(
    program: Program,
    train: list[Instance],
    reference_costs: dict[str, int],
    *,
    cache: EvaluationCache | None = None,
    versions: EvaluationVersions | None = None,
) -> EvalResult:
    for instance in train:
        if instance.split != "train":
            raise SplitAccessError(
                f"training fitness cannot access split {instance.split}"
            )
    _validate_inputs(train, reference_costs)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    ratios: list[Fraction] = []
    for instance in train:
        result = evaluate_program(
            program, instance, cache=cache, versions=versions
        )
        row = _evaluation_row(
            program, instance, result, reference_costs[instance.instance_id]
        )
        rows.append(row)
        if result.success:
            ratios.append(row["ratio"])
        else:
            failures.append(
                {
                    "instance_id": instance.instance_id,
                    "failure_reason": result.failure_reason,
                    "operational_cost": result.operational_cost,
                }
            )
    valid = not failures
    mean_ratio = sum(ratios, start=Fraction(0)) / len(ratios) if valid else None
    return EvalResult(
        valid_on_all_instances=valid,
        mean_ratio=mean_ratio,
        per_instance=rows,
        failures=failures,
    )


def benchmark(
    program: Program,
    instances: list[Instance],
    *,
    cache: EvaluationCache | None = None,
    versions: EvaluationVersions | None = None,
) -> list[SolveResult]:
    return [
        evaluate_program(program, instance, cache=cache, versions=versions)
        for instance in instances
    ]


def build_benchmark_rows(
    program: Program,
    instances: list[Instance],
    reference_costs: dict[str, int],
    *,
    cache: EvaluationCache | None = None,
    versions: EvaluationVersions | None = None,
) -> list[dict[str, Any]]:
    """Return one row per input, including failed rows with operational ratios."""
    _validate_inputs(instances, reference_costs)
    results = benchmark(program, instances, cache=cache, versions=versions)
    return [
        _evaluation_row(
            program,
            instance,
            result,
            reference_costs[instance.instance_id],
        )
        for instance, result in zip(instances, results)
    ]
