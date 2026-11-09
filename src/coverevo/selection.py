"""Validation-only finalist selection and manifest-gated held-out evaluation."""

from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .contracts import (
    EvalResult,
    Instance,
    OracleResult,
    Program,
    RunState,
    SelectionManifest,
)
from .evaluator import build_benchmark_rows
from .solver import greedy_reference
from .storage import dump_json_atomic, load_json_lossless


class SelectionError(RuntimeError):
    pass


class HeldOutAccessError(SelectionError):
    pass


def dataset_identity(instances: list[Instance]) -> str:
    """Hash identifiers and authoritative content hashes in stable order."""
    if not instances:
        raise ValueError("dataset identity requires at least one instance")
    identifiers = [instance.instance_id for instance in instances]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("dataset identity requires unique instance ids")
    payload = [
        {
            "instance_id": instance.instance_id,
            "base_graph_id": instance.base_graph_id,
            "split": instance.split,
            "variant": instance.variant,
            "content_hash": instance.content_hash,
        }
        for instance in sorted(instances, key=lambda item: item.instance_id)
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _reference_costs(instances: list[Instance]) -> dict[str, int]:
    result: dict[str, int] = {}
    for instance in instances:
        reference = greedy_reference(instance)
        if not reference.success or reference.raw_cost is None:
            raise SelectionError("registered greedy reference failed")
        result[instance.instance_id] = reference.raw_cost
    return result


def _validation_evaluation(
    program: Program,
    validation: list[Instance],
    references: dict[str, int],
) -> EvalResult:
    rows = build_benchmark_rows(program, validation, references)
    failures = [
        {
            "instance_id": row["instance_id"],
            "failure_reason": row["failure_reason"],
            "operational_cost": row["operational_cost"],
        }
        for row in rows
        if not row["success"]
    ]
    valid = not failures
    mean_ratio = (
        sum((row["ratio"] for row in rows), start=Fraction(0)) / len(rows)
        if valid
        else None
    )
    return EvalResult(valid, mean_ratio, rows, failures)


def _train_ranking(state: RunState) -> list[Program]:
    candidates: list[Program] = []
    for program_hash, program in state.programs.items():
        evaluation = state.evaluations.get(program_hash)
        if (
            evaluation is not None
            and evaluation.valid_on_all_instances
            and evaluation.mean_ratio is not None
        ):
            candidates.append(program)
    candidates.sort(
        key=lambda program: (
            state.evaluations[program.program_hash].mean_ratio,
            program.ast_nodes,
            program.program_hash,
        )
    )
    return candidates


def _registered_seed(state: RunState) -> Program:
    seeds = [
        program
        for program in state.programs.values()
        if program.provenance == "handwritten_seed"
    ]
    if len(seeds) != 1:
        raise SelectionError("run must contain exactly one registered seed")
    return seeds[0]


def select_finalists(
    states: list[RunState],
    validation: list[Instance],
    k: int = 3,
    *,
    experiment_hash: str,
    train_hash: str,
    code_commit: str,
    locked_at: str,
    artifact_class: str,
    config_hash: str,
    prompt_hash: str,
    model_snapshot_hash: str,
    expected_run_keys: Sequence[str],
) -> SelectionManifest:
    """Choose one validation winner per run from its training-only top-k."""
    if not states:
        raise ValueError("selection requires at least one completed run")
    if type(k) is not int or k <= 0:
        raise ValueError("k must be a positive integer")
    if not validation or any(instance.split != "validation" for instance in validation):
        raise ValueError("selection may access validation instances only")
    if len(experiment_hash) != 64 or len(train_hash) != 64:
        raise ValueError("experiment_hash and train_hash must be SHA-256 values")
    normalized_class = artifact_class.upper()
    if normalized_class not in {"FIXTURE", "MOCK", "LIVE"}:
        raise ValueError("selection artifact_class is invalid")
    run_keys = [f"{state.method}:{state.run_id}" for state in states]
    if len(run_keys) != len(set(run_keys)):
        raise SelectionError("selection states contain a duplicate method/run")
    if set(run_keys) != set(expected_run_keys) or len(expected_run_keys) != len(run_keys):
        raise SelectionError("selection is missing or has unexpected declared runs")
    if any(state.target_slots is None or state.target_slots <= 0
           or state.slots_completed != state.target_slots for state in states):
        raise SelectionError("all declared runs must be complete before validation selection")
    if any(record.cost_status=='UNKNOWN_BILLING' or record.candidate_status=='BILLING_PENDING'
           for state in states for record in state.proposals):
        raise SelectionError('all billing must be resolved before validation selection')
    references = _reference_costs(validation)
    shortlists: dict[str, Any] = {}
    winners: dict[str, Any] = {}
    for state, run_key in sorted(zip(states, run_keys), key=lambda item: item[1]):
        shortlist = _train_ranking(state)[:k]
        seed = _registered_seed(state)
        records: list[dict[str, Any]] = []
        valid_candidates: list[tuple[Fraction, int, str, Program]] = []
        for program in shortlist:
            evaluation = _validation_evaluation(program, validation, references)
            record = {
                "program_hash": program.program_hash,
                "training_mean_ratio": state.evaluations[
                    program.program_hash
                ].mean_ratio,
                "validation_status": (
                    "VALID" if evaluation.valid_on_all_instances else "FAILED"
                ),
                "validation_mean_ratio": evaluation.mean_ratio,
                "validation_failures": evaluation.failures,
            }
            records.append(record)
            if evaluation.valid_on_all_instances and evaluation.mean_ratio is not None:
                valid_candidates.append(
                    (
                        evaluation.mean_ratio,
                        program.ast_nodes,
                        program.program_hash,
                        program,
                    )
                )
        shortlists[run_key] = records
        if valid_candidates:
            ratio, _, _, winner = min(valid_candidates)
            winners[run_key] = {
                "program_hash": winner.program_hash,
                "validation_mean_ratio": ratio,
                "reason": "VALIDATION_SELECTED",
            }
        else:
            seed_evaluation = _validation_evaluation(seed, validation, references)
            if (
                not seed_evaluation.valid_on_all_instances
                or seed_evaluation.mean_ratio is None
            ):
                raise SelectionError("registered seed failed on validation")
            winners[run_key] = {
                "program_hash": seed.program_hash,
                "validation_mean_ratio": seed_evaluation.mean_ratio,
                "reason": "VALIDATION_ALL_FAILED_FALLBACK_SEED",
            }
    return SelectionManifest(
        experiment_hash=experiment_hash,
        per_run_shortlists=shortlists,
        per_run_winners=winners,
        train_hash=train_hash,
        validation_hash=dataset_identity(validation),
        code_commit=code_commit,
        locked_at=locked_at,
        config_hash=config_hash,
        prompt_hash=prompt_hash,
        model_snapshot_hash=model_snapshot_hash,
        artifact_class=normalized_class,
    )


def selection_manifest_to_mapping(manifest: SelectionManifest) -> dict[str, Any]:
    return asdict(manifest)


def _manifest_from_mapping(data: Mapping[str, Any]) -> SelectionManifest:
    return SelectionManifest(
        experiment_hash=data["experiment_hash"],
        per_run_shortlists=dict(data["per_run_shortlists"]),
        per_run_winners=dict(data["per_run_winners"]),
        train_hash=data["train_hash"],
        validation_hash=data["validation_hash"],
        code_commit=data["code_commit"],
        locked_at=data["locked_at"],
        config_hash=data.get("config_hash", ""),
        prompt_hash=data.get("prompt_hash", ""),
        model_snapshot_hash=data.get("model_snapshot_hash", ""),
        artifact_class=data.get("artifact_class", "FIXTURE"),
    )


def lock_selection_manifest(
    path: str | Path, manifest: SelectionManifest
) -> Path:
    """Create once; an exact replay is idempotent, any changed lock is rejected."""
    destination = Path(path)
    expected = selection_manifest_to_mapping(manifest)
    if destination.exists():
        observed = load_json_lossless(destination)
        if observed != expected:
            raise SelectionError("selection manifest is immutable")
        return destination
    dump_json_atomic(destination, expected)
    return destination


def load_selection_manifest(
    path: str | Path, *, expected_experiment_hash: str
) -> SelectionManifest:
    source = Path(path)
    if not source.is_file():
        raise HeldOutAccessError("locked selection manifest is required")
    raw = load_json_lossless(source)
    if not isinstance(raw, dict):
        raise HeldOutAccessError("selection manifest is malformed")
    manifest = _manifest_from_mapping(raw)
    if manifest.experiment_hash != expected_experiment_hash:
        raise HeldOutAccessError("selection manifest experiment hash mismatch")
    if not manifest.per_run_winners:
        raise HeldOutAccessError("selection manifest has no winners")
    return manifest


def evaluate_locked_winners(
    manifest_path: str | Path,
    program_registry: Mapping[str, Program],
    instances: list[Instance],
    reference_costs: dict[str, int],
    *,
    expected_experiment_hash: str,
    expected_instance_count: int,
    oracle_results: Mapping[str, OracleResult] | None = None,
    allow_research_test: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate all winner-instance pairs only after a matching immutable lock."""
    manifest = load_selection_manifest(
        manifest_path, expected_experiment_hash=expected_experiment_hash
    )
    if manifest.artifact_class == "LIVE" and not allow_research_test:
        raise HeldOutAccessError("research test access requires explicit approval")
    if manifest.artifact_class != "LIVE" and any(
        instance.metadata.get("artifact_class") not in {"FIXTURE", "MOCK"}
        for instance in instances
    ):
        raise HeldOutAccessError("MOCK selection may evaluate FIXTURE/MOCK instances only")
    if len(instances) != expected_instance_count:
        raise HeldOutAccessError("held-out instance count does not match protocol")
    if not instances or any(item.split not in {"test_id", "test_size", "test_topology"} for item in instances):
        raise HeldOutAccessError("held-out evaluation accepts test splits only")
    identifiers = [instance.instance_id for instance in instances]
    if len(identifiers) != len(set(identifiers)):
        raise HeldOutAccessError("held-out instance ids must be unique")
    rows: list[dict[str, Any]] = []
    for selection_key, winner in sorted(manifest.per_run_winners.items()):
        program_hash = winner.get("program_hash")
        program = program_registry.get(program_hash)
        if program is None:
            raise HeldOutAccessError(
                f"winner program is absent from registry: {program_hash}"
            )
        winner_rows = build_benchmark_rows(program, instances, reference_costs)
        for row in winner_rows:
            output = dict(row)
            output["selection_key"] = selection_key
            oracle = (
                None
                if oracle_results is None
                else oracle_results.get(row["instance_id"])
            )
            if oracle is None:
                output.update(
                    {
                        "oracle_status": None,
                        "oracle_incumbent_cost": None,
                        "oracle_lower_bound": None,
                        "oracle_certified_optimal": False,
                    }
                )
            else:
                output.update(
                    {
                        "oracle_status": oracle.status,
                        "oracle_incumbent_cost": oracle.incumbent_cost,
                        "oracle_lower_bound": oracle.lower_bound,
                        "oracle_certified_optimal": (
                            oracle.status == "OPTIMAL" and oracle.certified_optimal
                        ),
                    }
                )
            rows.append(output)
    expected_rows = len(manifest.per_run_winners) * len(instances)
    if len(rows) != expected_rows:
        raise HeldOutAccessError("held-out evaluation is missing a winner-instance pair")
    return rows


def summarize_oracle_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report certification scope without promoting a subset to the full test set."""
    if not rows:
        raise ValueError("oracle coverage requires benchmark rows")
    by_instance: dict[str, bool] = {}
    for row in rows:
        identifier = row.get("instance_id")
        if not isinstance(identifier, str):
            raise ValueError("benchmark row is missing instance_id")
        certified = row.get("oracle_certified_optimal") is True
        if identifier in by_instance and by_instance[identifier] != certified:
            raise ValueError("oracle certification differs across winner rows")
        by_instance[identifier] = certified
    certified_count = sum(by_instance.values())
    full = certified_count == len(by_instance)
    return {
        "total_instances": len(by_instance),
        "certified_instances": certified_count,
        "full_test_certified": full,
        "scope": "FULL_TEST" if full else "CERTIFIED_SUBSET",
    }
