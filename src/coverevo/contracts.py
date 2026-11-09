"""Shared semantic contracts used across COVER-EVO modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class Instance:
    instance_id: str
    base_graph_id: str
    split: str
    n: int
    edges: tuple[tuple[int, int], ...]
    costs: tuple[int, ...]
    variant: str
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Instance":
        """Restore tuple-valued fields from a JSON-compatible mapping."""
        return cls(
            instance_id=str(data["instance_id"]),
            base_graph_id=str(data["base_graph_id"]),
            split=str(data["split"]),
            n=data["n"],
            edges=tuple(tuple(edge) for edge in data["edges"]),
            costs=tuple(data["costs"]),
            variant=str(data["variant"]),
            content_hash=str(data["content_hash"]),
            metadata=dict(data.get("metadata", {})),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return a lossless JSON-compatible representation."""
        return {
            "instance_id": self.instance_id,
            "base_graph_id": self.base_graph_id,
            "split": self.split,
            "n": self.n,
            "edges": [list(edge) for edge in self.edges],
            "costs": list(self.costs),
            "variant": self.variant,
            "content_hash": self.content_hash,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class FeatureVector:
    gain: Fraction
    cost: Fraction
    rarity: Fraction
    overlap: Fraction
    remaining: Fraction


@dataclass(frozen=True, slots=True)
class Program:
    expression: str
    canonical_expression: str
    program_hash: str
    ast_nodes: int
    ast_depth: int
    provenance: str
    rationale: str = ""


@dataclass(frozen=True, slots=True)
class SolveResult:
    success: bool
    selected_before_pruning: tuple[int, ...]
    selected: tuple[int, ...]
    cost_before_pruning: int | None
    raw_cost: int | None
    operational_cost: int
    construction_steps: int
    failure_reason: str | None
    runtime_ns: int
    trace: list[dict[str, Any]] | None


@dataclass(frozen=True, slots=True)
class OracleResult:
    status: str
    selected: tuple[int, ...] | None
    incumbent_cost: int | None
    lower_bound: float | None
    certified_optimal: bool
    wall_time_s: float
    solver_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvalResult:
    valid_on_all_instances: bool
    mean_ratio: Fraction | None
    per_instance: list[dict[str, Any]]
    failures: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ProposalRecord:
    experiment_id: str
    method: str
    run_id: int
    slot: int
    prompt_hash: str
    response_id: str | None
    program_hash: str | None
    request_status: str
    candidate_status: str
    cost_status: str
    billed_cost: Decimal | None
    usage: dict[str, Any]
    parent_hashes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SelectionManifest:
    experiment_hash: str
    per_run_shortlists: dict[str, Any]
    per_run_winners: dict[str, Any]
    train_hash: str
    validation_hash: str
    code_commit: str
    locked_at: str
    config_hash: str = ""
    prompt_hash: str = ""
    model_snapshot_hash: str = ""
    artifact_class: str = "FIXTURE"


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    response_id: str | None
    content: str | None
    request_status: str
    cost_status: str
    billed_cost: Decimal | None
    usage: dict[str, Any]
    finish_reason: str | None = None
    actual_model: str | None = None
    actual_provider: str | None = None
    artifact_class: str = "MOCK"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Frozen provider request settings after a successful compatibility check."""

    model_id: str
    canonical_slug_verified: str | None
    provider_slug_verified: str | None
    temperature: float
    max_tokens: int
    reasoning_effort: str
    response_format: str
    stream: bool
    allow_model_fallbacks: bool
    allow_provider_fallbacks: bool
    require_parameters: bool
    capabilities_snapshot: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ModelConfig":
        return cls(
            model_id=str(data["id"]),
            canonical_slug_verified=(
                None
                if data.get("canonical_slug_verified") is None
                else str(data["canonical_slug_verified"])
            ),
            provider_slug_verified=(
                None
                if data.get("provider_slug_verified") is None
                else str(data["provider_slug_verified"])
            ),
            temperature=float(data["temperature"]),
            max_tokens=data["max_tokens"],
            reasoning_effort=str(data["reasoning_effort"]),
            response_format=str(data["response_format"]),
            stream=data["stream"],
            allow_model_fallbacks=data["allow_model_fallbacks"],
            allow_provider_fallbacks=data["allow_provider_fallbacks"],
            require_parameters=data["require_parameters"],
            capabilities_snapshot=(
                None
                if data.get("capabilities_snapshot") is None
                else str(data["capabilities_snapshot"])
            ),
        )


@dataclass(frozen=True, slots=True)
class ArchiveEntry:
    program: Program
    evaluation: EvalResult
    behavior_hash: str
    source_slot: int


@dataclass(slots=True)
class RunState:
    experiment_id: str
    method: str
    run_id: int
    mode: str
    slots_completed: int
    archive: list[ArchiveEntry]
    programs: dict[str, Program]
    evaluations: dict[str, EvalResult]
    proposals: list[ProposalRecord]
    raw_responses: list[dict[str, Any]]
    prompt_records: list[dict[str, Any]]
    syntactic_unique_count: int
    behavior_unique_count: int
    rng_state: Any | None = None
    target_slots: int | None = None
    pending_rng_state: Any | None = None


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    protocol_version: str
    seed_namespace: str
    mode: str
    allow_live: bool
    problem: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    solver: dict[str, Any] = field(default_factory=dict)
    dsl: dict[str, Any] = field(default_factory=dict)
    search: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)
    oracle: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ExperimentConfig":
        return cls(
            protocol_version=str(data["protocol_version"]),
            seed_namespace=str(data["seed_namespace"]),
            mode=str(data["mode"]),
            allow_live=data["allow_live"],
            problem=dict(data.get("problem", {})),
            data=dict(data.get("data", {})),
            solver=dict(data.get("solver", {})),
            dsl=dict(data.get("dsl", {})),
            search=dict(data.get("search", {})),
            model=dict(data.get("model", {})),
            budget=dict(data.get("budget", {})),
            oracle=dict(data.get("oracle", {})),
            analysis=dict(data.get("analysis", {})),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "seed_namespace": self.seed_namespace,
            "mode": self.mode,
            "allow_live": self.allow_live,
            "problem": dict(self.problem),
            "data": dict(self.data),
            "solver": dict(self.solver),
            "dsl": dict(self.dsl),
            "search": dict(self.search),
            "model": dict(self.model),
            "budget": dict(self.budget),
            "oracle": dict(self.oracle),
            "analysis": dict(self.analysis),
        }
