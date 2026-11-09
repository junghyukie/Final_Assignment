"""Deterministic MWDS constructors, common pruning and independent checking."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from time import perf_counter_ns
from typing import Any

from .contracts import FeatureVector, Instance, Program, SolveResult
from .dsl import NumericLimitError, parse_program, score_program
from .features import GraphContext, build_context, compute_features
from .instances import validate_instance


CandidateScorer = Callable[[int, int, FeatureVector], Fraction]


def _closed_neighborhoods(instance: Instance) -> tuple[frozenset[int], ...]:
    neighborhoods = [set((node,)) for node in range(instance.n)]
    for u, v in instance.edges:
        neighborhoods[u].add(v)
        neighborhoods[v].add(u)
    return tuple(frozenset(neighborhood) for neighborhood in neighborhoods)


def _valid_selected(instance: Instance, selected: tuple[int, ...]) -> bool:
    return (
        len(selected) == len(set(selected))
        and all(type(node) is int and 0 <= node < instance.n for node in selected)
    )


def check_solution(instance: Instance, selected: tuple[int, ...]) -> bool:
    """Recompute feasibility solely from original nodes and edges."""
    validate_instance(instance)
    if not _valid_selected(instance, selected):
        return False
    covered: set[int] = set()
    neighborhoods = _closed_neighborhoods(instance)
    for node in selected:
        covered.update(neighborhoods[node])
    return len(covered) == instance.n


def _prune_with_removed(
    instance: Instance, selected: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    validate_instance(instance)
    if not _valid_selected(instance, selected):
        raise ValueError("selected nodes must be unique in-range integers")
    if not check_solution(instance, selected):
        raise ValueError("cannot prune an infeasible selected set")

    neighborhoods = _closed_neighborhoods(instance)
    coverage_count = [0] * instance.n
    retained = set(selected)
    for node in selected:
        for covered in neighborhoods[node]:
            coverage_count[covered] += 1

    pruning_order = tuple(
        sorted(selected, key=lambda node: (-instance.costs[node], node))
    )
    removed: list[int] = []
    for node in pruning_order:
        if all(coverage_count[covered] >= 2 for covered in neighborhoods[node]):
            retained.remove(node)
            removed.append(node)
            for covered in neighborhoods[node]:
                coverage_count[covered] -= 1

    pruned = tuple(node for node in selected if node in retained)
    if not check_solution(instance, pruned):
        raise RuntimeError("pruning violated feasibility")
    return pruned, tuple(removed), pruning_order


def prune_solution(instance: Instance, selected: tuple[int, ...]) -> tuple[int, ...]:
    """Apply the single locked reverse-delete pass and preserve input order."""
    pruned, _, _ = _prune_with_removed(instance, selected)
    return pruned


def _feature_mapping(features: FeatureVector) -> dict[str, Fraction]:
    return {
        "gain": features.gain,
        "cost": features.cost,
        "rarity": features.rarity,
        "overlap": features.overlap,
        "remaining": features.remaining,
    }


def _failure_result(
    instance: Instance,
    selected: list[int],
    started_ns: int,
    trace_events: list[dict[str, Any]] | None,
    *,
    reason: str,
    failed_node: int | None,
) -> SolveResult:
    if trace_events is not None:
        trace_events.append(
            {"phase": "failure", "reason": reason, "candidate_node": failed_node}
        )
    return SolveResult(
        success=False,
        selected_before_pruning=tuple(selected),
        selected=(),
        cost_before_pruning=None,
        raw_cost=None,
        operational_cost=sum(instance.costs),
        construction_steps=len(selected),
        failure_reason=reason,
        runtime_ns=perf_counter_ns() - started_ns,
        trace=trace_events,
    )


def _construct(
    instance: Instance,
    scorer: CandidateScorer,
    *,
    prune: bool,
    trace: bool,
) -> SolveResult:
    started_ns = perf_counter_ns()
    validate_instance(instance)
    context: GraphContext = build_context(instance)
    selected: list[int] = []
    selected_set: set[int] = set()
    uncovered: set[int] = set(range(instance.n))
    trace_events: list[dict[str, Any]] | None = [] if trace else None

    while uncovered:
        candidates: list[dict[str, Any]] = []
        for node in range(instance.n):
            if node in selected_set:
                continue
            newly_covered = context.closed_neighborhoods[node].intersection(uncovered)
            gain_count = len(newly_covered)
            if gain_count == 0:
                continue
            features = compute_features(context, frozenset(uncovered), node)
            try:
                score = scorer(node, gain_count, features)
            except NumericLimitError:
                return _failure_result(
                    instance,
                    selected,
                    started_ns,
                    trace_events,
                    reason="NUMERIC_LIMIT",
                    failed_node=node,
                )
            candidates.append(
                {
                    "node": node,
                    "gain_count": gain_count,
                    "already_covered": node not in uncovered,
                    "features": _feature_mapping(features),
                    "score": score,
                }
            )

        if not candidates:
            raise RuntimeError("non-empty uncovered set has no positive-gain candidate")
        chosen_record = max(
            candidates,
            key=lambda item: (
                item["score"],
                -instance.costs[item["node"]],
                -item["node"],
            ),
        )
        chosen = chosen_record["node"]
        if trace_events is not None:
            trace_events.append(
                {
                    "phase": "construct",
                    "step": len(selected),
                    "selected_before": tuple(selected),
                    "uncovered_before": tuple(sorted(uncovered)),
                    "candidates": candidates,
                    "chosen": chosen,
                }
            )
        selected.append(chosen)
        selected_set.add(chosen)
        uncovered.difference_update(context.closed_neighborhoods[chosen])

    selected_before = tuple(selected)
    if not check_solution(instance, selected_before):
        raise RuntimeError("constructor returned an infeasible solution")
    cost_before = sum(instance.costs[node] for node in selected_before)
    removed: tuple[int, ...] = ()
    pruning_order: tuple[int, ...] = ()
    if prune:
        final_selected, removed, pruning_order = _prune_with_removed(
            instance, selected_before
        )
    else:
        final_selected = selected_before
    raw_cost = sum(instance.costs[node] for node in final_selected)
    if trace_events is not None:
        trace_events.append(
            {
                "phase": "pruning",
                "order": pruning_order,
                "removed": removed,
                "selected_after": final_selected,
            }
        )
    return SolveResult(
        success=True,
        selected_before_pruning=selected_before,
        selected=final_selected,
        cost_before_pruning=cost_before,
        raw_cost=raw_cost,
        operational_cost=raw_cost,
        construction_steps=len(selected_before),
        failure_reason=None,
        runtime_ns=perf_counter_ns() - started_ns,
        trace=trace_events,
    )


def greedy_reference(
    instance: Instance, *, prune: bool = True, trace: bool = False
) -> SolveResult:
    """Run exact gain-count/original-cost greedy followed by common pruning."""

    def score(node: int, gain_count: int, features: FeatureVector) -> Fraction:
        del features
        return Fraction(gain_count, instance.costs[node])

    return _construct(instance, score, prune=prune, trace=trace)


def solve(
    instance: Instance,
    program: Program,
    *,
    prune: bool = True,
    trace: bool = False,
) -> SolveResult:
    """Run a validated DSL scoring program in the fixed constructor."""

    def score(node: int, gain_count: int, features: FeatureVector) -> Fraction:
        del node, gain_count
        return score_program(program, features)

    return _construct(instance, score, prune=prune, trace=trace)


def rarity_reference(
    instance: Instance, *, prune: bool = True, trace: bool = False
) -> SolveResult:
    """Run the registered rarity-weighted handwritten reference (RGR+P)."""
    program = parse_program(
        "div(gain * rarity, cost)", provenance="handwritten"
    )
    return solve(instance, program, prune=prune, trace=trace)


def reverse_delete(instance: Instance) -> SolveResult:
    """Start from all vertices and apply the common pruning order."""
    started_ns = perf_counter_ns()
    validate_instance(instance)
    selected_before = tuple(range(instance.n))
    selected, removed, pruning_order = _prune_with_removed(instance, selected_before)
    raw_cost = sum(instance.costs[node] for node in selected)
    return SolveResult(
        success=True,
        selected_before_pruning=selected_before,
        selected=selected,
        cost_before_pruning=sum(instance.costs),
        raw_cost=raw_cost,
        operational_cost=raw_cost,
        construction_steps=0,
        failure_reason=None,
        runtime_ns=perf_counter_ns() - started_ns,
        trace=[
            {
                "phase": "pruning",
                "order": pruning_order,
                "removed": removed,
                "selected_after": selected,
            }
        ],
    )
