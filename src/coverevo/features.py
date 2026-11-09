"""Exact, leakage-free feature computation for MWDS candidates."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .contracts import FeatureVector, Instance
from .instances import validate_instance


class FeatureComputationError(ValueError):
    """Raised when features are requested outside the constructor contract."""


@dataclass(frozen=True, slots=True)
class GraphContext:
    n: int
    costs: tuple[int, ...]
    closed_neighborhoods: tuple[frozenset[int], ...]
    mean_cost: Fraction
    rarity_by_demand_node: tuple[Fraction, ...]


def build_context(instance: Instance) -> GraphContext:
    """Precompute graph-only values without carrying split or metadata fields."""
    validate_instance(instance)
    neighborhoods = [set((node,)) for node in range(instance.n)]
    for u, v in instance.edges:
        neighborhoods[u].add(v)
        neighborhoods[v].add(u)
    closed = tuple(frozenset(neighborhood) for neighborhood in neighborhoods)
    return GraphContext(
        n=instance.n,
        costs=instance.costs,
        closed_neighborhoods=closed,
        mean_cost=Fraction(sum(instance.costs), instance.n),
        rarity_by_demand_node=tuple(
            Fraction(1, len(neighborhood)) for neighborhood in closed
        ),
    )


def compute_features(
    context: GraphContext, uncovered: frozenset[int], v: int
) -> FeatureVector:
    """Compute the five exact features for a positive-gain candidate node."""
    if not uncovered:
        raise FeatureComputationError("uncovered must be non-empty")
    if any(type(node) is not int or not 0 <= node < context.n for node in uncovered):
        raise FeatureComputationError("uncovered contains an invalid node")
    if type(v) is not int or not 0 <= v < context.n:
        raise FeatureComputationError("candidate node is outside graph range")

    newly_covered = context.closed_neighborhoods[v].intersection(uncovered)
    gain_count = len(newly_covered)
    if gain_count == 0:
        raise FeatureComputationError("candidate must have positive gain")

    return FeatureVector(
        gain=Fraction(gain_count, len(uncovered)),
        cost=Fraction(context.costs[v], 1) / context.mean_cost,
        rarity=sum(
            (context.rarity_by_demand_node[node] for node in newly_covered),
            start=Fraction(0),
        )
        / gain_count,
        overlap=Fraction(1) - Fraction(
            gain_count, len(context.closed_neighborhoods[v])
        ),
        remaining=Fraction(len(uncovered), context.n),
    )
