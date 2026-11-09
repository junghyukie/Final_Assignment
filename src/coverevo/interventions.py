"""Frozen-program feature and pruning interventions."""
from __future__ import annotations
from dataclasses import replace
import re
from typing import Any
from .contracts import Instance, Program
from .dsl import parse_program
from .solver import solve


def intervene_program(program: Program, feature: str) -> Program:
    """Mechanically replace one used feature with zero; never optimize the result."""
    allowed = {"gain", "cost", "rarity", "overlap", "remaining"}
    if feature not in allowed:
        raise ValueError("feature is not part of the COVER-EVO DSL")
    pattern = re.compile(rf"\b{re.escape(feature)}\b")
    if pattern.search(program.expression) is None:
        raise ValueError(f"program does not use feature {feature}")
    expression = pattern.sub("0", program.expression)
    changed = parse_program(expression, "handwritten")
    return replace(
        changed,
        rationale=f"Mechanical intervention: {feature}=0 in {program.program_hash}.",
    )


def build_ablation_rows(
    program: Program,
    instances: list[Instance],
    *,
    feature: str,
) -> list[dict[str, Any]]:
    """Run locked, no-pruning and feature-zero variants without model calls."""
    intervened = intervene_program(program, feature)
    variants = (
        ("LOCKED", program, True),
        ("NO_PRUNING", program, False),
        (f"{feature.upper()}_ZERO", intervened, True),
    )
    rows: list[dict[str, Any]] = []
    for label, active_program, prune in variants:
        for instance in instances:
            result = solve(instance, active_program, prune=prune, trace=True)
            rows.append(
                {
                    "artifact_class": "EXPLORATORY",
                    "ablation": label,
                    "source_program_hash": program.program_hash,
                    "program_hash": active_program.program_hash,
                    "instance_id": instance.instance_id,
                    "base_graph_id": instance.base_graph_id,
                    "success": result.success,
                    "selected_before_pruning": result.selected_before_pruning,
                    "selected": result.selected,
                    "raw_cost": result.raw_cost,
                    "operational_cost": result.operational_cost,
                    "failure_reason": result.failure_reason,
                    "trace": result.trace,
                }
            )
    return rows
