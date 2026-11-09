"""Deterministic instance validation, graph generation and cost assignment."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import random
import string
from typing import Any, Iterable, Sequence

import networkx as nx

from .contracts import ExperimentConfig, Instance


class InstanceValidationError(ValueError):
    """Raised when an instance violates the simple undirected MWDS contract."""


_HEX = frozenset(string.hexdigits)
_SPLITS = (
    "pilot",
    "train",
    "validation",
    "test_id",
    "test_size",
    "test_topology",
)


def _json_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def graph_hash(n: int, edges: Sequence[tuple[int, int]]) -> str:
    return _json_hash({"n": n, "edges": [list(edge) for edge in edges]})


def instance_content_hash(
    n: int, edges: Sequence[tuple[int, int]], costs: Sequence[int]
) -> str:
    return _json_hash(
        {
            "n": n,
            "edges": [list(edge) for edge in edges],
            "costs": list(costs),
        }
    )


def validate_instance(instance: Instance) -> None:
    """Validate the canonical simple-undirected graph and positive costs."""
    if type(instance.n) is not int or instance.n < 1:
        raise InstanceValidationError("n must be at least 1")
    if len(instance.costs) != instance.n:
        raise InstanceValidationError("cost length must equal n")
    if any(type(cost) is not int or cost <= 0 for cost in instance.costs):
        raise InstanceValidationError("costs must be positive integer values")

    previous: tuple[int, int] | None = None
    seen: set[tuple[int, int]] = set()
    for edge in instance.edges:
        if len(edge) != 2 or any(type(endpoint) is not int for endpoint in edge):
            raise InstanceValidationError("each edge must have two integer endpoints")
        u, v = edge
        if not (0 <= u < instance.n and 0 <= v < instance.n):
            raise InstanceValidationError("edge endpoint is outside node range")
        if u == v:
            raise InstanceValidationError("self-loop is not allowed")
        if u > v:
            raise InstanceValidationError("edges must use canonical undirected order u < v")
        if edge in seen:
            raise InstanceValidationError("duplicate edge is not allowed")
        if previous is not None and edge < previous:
            raise InstanceValidationError("edges must be sorted lexicographically")
        seen.add(edge)
        previous = edge

    if (
        len(instance.content_hash) != 64
        or any(character not in _HEX for character in instance.content_hash)
    ):
        raise InstanceValidationError("content_hash must be a 64-character SHA-256 hex")
    if instance.metadata.get("hash_algorithm") == "sha256:instance-v1":
        expected = instance_content_hash(instance.n, instance.edges, instance.costs)
        if instance.content_hash != expected:
            raise InstanceValidationError("content_hash does not match instance content")


def derive_seed(namespace: str, split: str, index: int, purpose: str) -> int:
    """Derive a stable 32-bit seed from the protocol recipe."""
    material = f"{namespace}|{split}|{index}|{purpose}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def _subseed(seed: int, purpose: str) -> int:
    material = f"{seed}|{purpose}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def _validate_generator_parameters(n: int, m: int) -> None:
    if type(n) is not int or type(m) is not int:
        raise ValueError("n and m must be integers")
    if n < 2:
        raise ValueError("n must be at least 2 for graph generation")
    if not 1 <= m < n:
        raise ValueError("m must satisfy 1 <= m < n")
    edge_budget = m * (n - m)
    if edge_budget > n * (n - 1) // 2:
        raise ValueError("m * (n - m) exceeds the number of possible edges")


def _relabel(
    n: int,
    edges: Iterable[tuple[int, int]],
    positions: Sequence[tuple[float, float]] | None,
    seed: int,
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[float, float], ...] | None,
    tuple[int, ...],
]:
    new_label_for_old = list(range(n))
    random.Random(_subseed(seed, "labels")).shuffle(new_label_for_old)
    relabeled_edges = tuple(
        sorted(
            (min(new_label_for_old[u], new_label_for_old[v]),
             max(new_label_for_old[u], new_label_for_old[v]))
            for u, v in edges
        )
    )
    relabeled_positions: tuple[tuple[float, float], ...] | None = None
    if positions is not None:
        by_new_label: list[tuple[float, float] | None] = [None] * n
        for old, new in enumerate(new_label_for_old):
            by_new_label[new] = positions[old]
        relabeled_positions = tuple(position for position in by_new_label if position is not None)
    return relabeled_edges, relabeled_positions, tuple(new_label_for_old)


def _generated_instance(
    *,
    generator: str,
    n: int,
    m: int,
    seed: int,
    edges: tuple[tuple[int, int], ...],
    metadata: dict[str, Any],
) -> Instance:
    costs = (1,) * n
    topology_hash = graph_hash(n, edges)
    combined_metadata = {
        "artifact_class": "GENERATED_OFFLINE",
        "generator": generator,
        "generator_seed": seed,
        "attachment_m": m,
        "edge_budget": m * (n - m),
        "graph_hash": topology_hash,
        "temporary_unit_costs": True,
        "hash_algorithm": "sha256:instance-v1",
        **metadata,
    }
    instance = Instance(
        instance_id=f"unassigned-{generator}-{seed}",
        base_graph_id=f"unassigned-{topology_hash[:16]}",
        split="unassigned",
        n=n,
        edges=edges,
        costs=costs,
        variant="unit",
        content_hash=instance_content_hash(n, edges, costs),
        metadata=combined_metadata,
    )
    validate_instance(instance)
    return instance


def make_fixed_edge_geometric(n: int, m: int, seed: int) -> Instance:
    """Generate a geometric graph using exactly ``m * (n - m)`` nearest pairs."""
    _validate_generator_parameters(n, m)
    position_rng = random.Random(_subseed(seed, "positions"))
    positions = tuple((position_rng.random(), position_rng.random()) for _ in range(n))
    distances = []
    for u in range(n):
        x_u, y_u = positions[u]
        for v in range(u + 1, n):
            x_v, y_v = positions[v]
            squared_distance = (x_u - x_v) ** 2 + (y_u - y_v) ** 2
            distances.append((squared_distance, u, v))
    distances.sort()
    edge_budget = m * (n - m)
    chosen = tuple((u, v) for _, u, v in distances[:edge_budget])
    boundary_tie = (
        edge_budget < len(distances)
        and distances[edge_budget - 1][0] == distances[edge_budget][0]
    )
    edges, relabeled_positions, permutation = _relabel(n, chosen, positions, seed)
    return _generated_instance(
        generator="fixed_edge_geometric",
        n=n,
        m=m,
        seed=seed,
        edges=edges,
        metadata={
            "positions": [list(position) for position in relabeled_positions or ()],
            "old_to_new_permutation": list(permutation),
            "distance_tie_at_boundary": boundary_tie,
        },
    )


def make_ba_instance(n: int, m: int, seed: int) -> Instance:
    """Generate and independently relabel the pinned NetworkX BA recipe."""
    _validate_generator_parameters(n, m)
    graph = nx.barabasi_albert_graph(n, m, seed=_subseed(seed, "topology"))
    original_edges = tuple((min(u, v), max(u, v)) for u, v in graph.edges())
    edges, _, permutation = _relabel(n, original_edges, None, seed)
    return _generated_instance(
        generator="barabasi_albert",
        n=n,
        m=m,
        seed=seed,
        edges=edges,
        metadata={
            "networkx_version": nx.__version__,
            "initial_graph": "networkx_default_star_m_plus_one",
            "old_to_new_permutation": list(permutation),
        },
    )


def _degrees(instance: Instance) -> list[int]:
    result = [0] * instance.n
    for u, v in instance.edges:
        result[u] += 1
        result[v] += 1
    return result


def _average_ranks(values: Sequence[int]) -> list[float]:
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


def _spearman(values_a: Sequence[int], values_b: Sequence[int]) -> float | None:
    ranks_a = _average_ranks(values_a)
    ranks_b = _average_ranks(values_b)
    mean_a = sum(ranks_a) / len(ranks_a)
    mean_b = sum(ranks_b) / len(ranks_b)
    centered_a = [value - mean_a for value in ranks_a]
    centered_b = [value - mean_b for value in ranks_b]
    denominator = math.sqrt(
        sum(value * value for value in centered_a)
        * sum(value * value for value in centered_b)
    )
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(centered_a, centered_b)) / denominator


def assign_cost_variants(instance: Instance, seed: int) -> dict[str, Instance]:
    """Assign random, hub-expensive and hub-cheap permutations of one multiset."""
    validate_instance(instance)
    sorted_costs = sorted(instance.costs)
    degrees = _degrees(instance)

    random_costs = list(instance.costs)
    random.Random(_subseed(seed, "random_cost_order")).shuffle(random_costs)

    degree_order = list(range(instance.n))
    random.Random(_subseed(seed, "degree_ties")).shuffle(degree_order)
    degree_order.sort(key=lambda node: degrees[node])

    expensive = [0] * instance.n
    cheap = [0] * instance.n
    for position, node in enumerate(degree_order):
        expensive[node] = sorted_costs[position]
        cheap[node] = sorted_costs[-1 - position]

    topology_hash = graph_hash(instance.n, instance.edges)
    output: dict[str, Instance] = {}
    for name, costs_list in (
        ("random", random_costs),
        ("hub_expensive", expensive),
        ("hub_cheap", cheap),
    ):
        costs = tuple(costs_list)
        metadata = dict(instance.metadata)
        metadata.update(
            {
                "graph_hash": topology_hash,
                "cost_variant": name,
                "cost_order_seed": seed,
                "cost_degree_spearman": _spearman(degrees, costs),
                "temporary_unit_costs": False,
                "hash_algorithm": "sha256:instance-v1",
            }
        )
        variant = replace(
            instance,
            instance_id=f"{instance.base_graph_id}-{name}",
            variant=name,
            costs=costs,
            content_hash=instance_content_hash(instance.n, instance.edges, costs),
            metadata=metadata,
        )
        validate_instance(variant)
        output[name] = variant
    return output


def _required_int(data: dict[str, Any], name: str) -> int:
    value = data.get(name)
    if type(value) is not int:
        raise ValueError(f"data.{name} must be an integer")
    return value


def build_datasets(
    config: ExperimentConfig, *, splits: Iterable[str] | None = None
) -> dict[str, list[Instance]]:
    """Build only requested deterministic splits and reject topology reuse."""
    requested = tuple(splits) if splits is not None else _SPLITS
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("splits must be a non-empty sequence without duplicates")
    unknown = set(requested) - set(_SPLITS)
    if unknown:
        raise ValueError(f"unknown splits: {sorted(unknown)}")

    data = config.data
    m = _required_int(data, "attachment_m")
    cost_min = _required_int(data, "cost_min")
    cost_max = _required_int(data, "cost_max")
    if cost_min < 1 or cost_min > cost_max:
        raise ValueError("cost range must be positive and ordered")

    definitions = {
        "pilot": (_required_int(data, "pilot_graphs"), _required_int(data, "n_train"), "geometric"),
        "train": (_required_int(data, "train_graphs"), _required_int(data, "n_train"), "geometric"),
        "validation": (_required_int(data, "validation_graphs"), _required_int(data, "n_train"), "geometric"),
        "test_id": (_required_int(data, "test_id_graphs"), _required_int(data, "n_test_id"), "geometric"),
        "test_size": (_required_int(data, "test_size_graphs"), _required_int(data, "n_test_size"), "geometric"),
        "test_topology": (_required_int(data, "test_topology_graphs"), _required_int(data, "n_test_id"), "ba"),
    }
    allowed_variants = tuple(data.get("id_cost_variants", ("random", "hub_expensive", "hub_cheap")))
    if set(allowed_variants) - {"random", "hub_expensive", "hub_cheap"}:
        raise ValueError("id_cost_variants contains an unknown variant")

    datasets: dict[str, list[Instance]] = {}
    seen_graph_hashes: dict[str, str] = {}
    for split in requested:
        count, n, family = definitions[split]
        if count < 0:
            raise ValueError(f"graph count for {split} cannot be negative")
        items: list[Instance] = []
        for index in range(count):
            topology_seed = derive_seed(config.seed_namespace, split, index, "positions")
            generated = (
                make_ba_instance(n, m, topology_seed)
                if family == "ba"
                else make_fixed_edge_geometric(n, m, topology_seed)
            )
            base_graph_id = f"{split}-{index:04d}"
            topology_hash = generated.metadata["graph_hash"]
            if topology_hash in seen_graph_hashes:
                other = seen_graph_hashes[topology_hash]
                raise ValueError(f"structural duplicate graph: {other} and {base_graph_id}")
            seen_graph_hashes[topology_hash] = base_graph_id

            cost_seed = derive_seed(config.seed_namespace, split, index, "costs")
            cost_rng = random.Random(cost_seed)
            costs = tuple(cost_rng.randint(cost_min, cost_max) for _ in range(n))
            metadata = dict(generated.metadata)
            metadata.update(
                {
                    "artifact_class": "GENERATED_OFFLINE",
                    "split_index": index,
                    "topology_seed": topology_seed,
                    "cost_seed": cost_seed,
                }
            )
            base = replace(
                generated,
                instance_id=f"{base_graph_id}-base",
                base_graph_id=base_graph_id,
                split=split,
                costs=costs,
                variant="base",
                content_hash=instance_content_hash(n, generated.edges, costs),
                metadata=metadata,
            )
            variants = assign_cost_variants(
                base, derive_seed(config.seed_namespace, split, index, "cost_order")
            )
            names = allowed_variants if split == "test_id" else ("random",)
            for name in names:
                item = replace(
                    variants[name],
                    instance_id=f"{base_graph_id}-{name}",
                    base_graph_id=base_graph_id,
                    split=split,
                )
                validate_instance(item)
                items.append(item)
        datasets[split] = items
    return datasets


def serialize_dataset_manifest(datasets: dict[str, list[Instance]]) -> str:
    """Serialize materialized instances deterministically for optional live discovery."""
    payload = {
        "artifact_class": "GENERATED_OFFLINE",
        "splits": {
            split: [item.to_mapping() for item in sorted(items, key=lambda value: value.instance_id)]
            for split, items in sorted(datasets.items())
        },
    }
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
