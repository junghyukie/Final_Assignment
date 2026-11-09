"""Adapter for Jovanovic--Tuba--Simian's original T1/T2 MWDS files.

The diagonal encodes self-coverage, which Instance/solver already provide.
Vertex order and positive integer weights are never transformed.
"""
from __future__ import annotations

import re

from .contracts import Instance
from .instances import graph_hash, instance_content_hash, validate_instance


MEMBER = re.compile(r"(T[12])/V(\d+)E(\d+)/(\d+)/Test/Problem\.dat_(\d+)_(\d+)_(\d+)$")


def parse_original(raw: bytes, member: str) -> Instance:
    match = MEMBER.fullmatch(member)
    if match is None:
        raise ValueError(f"unrecognized original member: {member}")
    family, ns, ms, seed, file_n, file_m, file_seed = match.groups()
    if (ns, ms, seed) != (file_n, file_m, file_seed):
        raise ValueError("directory and filename disagree")
    lines = [line.strip() for line in raw.decode("ascii").splitlines() if line.strip()]
    n = int(ns)
    if (len(lines) != 3 * n + 5 or lines[0] != "NumberOfNodes:"
            or int(lines[1]) != n or lines[2] != "Positions"
            or "WEIGHTS" not in lines[n + 3]
            or "CONNECTIONS" not in lines[2 * n + 4]):
        raise ValueError("malformed original header or dimensions")
    # Positions are not used by any scoring feature, but retain their bytes in source.
    for line in lines[3:n + 3]:
        if len(line.split()) != 2:
            raise ValueError("malformed coordinate row")
        tuple(float(x) for x in line.split())
    costs = tuple(int(line) for line in lines[n + 4:2 * n + 4])
    if len(costs) != n or any(c <= 0 for c in costs):
        raise ValueError("weights must be positive integers")
    matrix = [tuple(int(x) for x in line.split()) for line in lines[2 * n + 5:]]
    if any(len(row) != n or any(v not in (0, 1) for v in row) for row in matrix):
        raise ValueError("adjacency must be a square binary matrix")
    if any(matrix[i][i] != 1 for i in range(n)):
        raise ValueError("original diagonal must encode self-coverage")
    if any(matrix[i][j] != matrix[j][i] for i in range(n) for j in range(i + 1, n)):
        raise ValueError("adjacency must be symmetric")
    edges = tuple((i, j) for i in range(n) for j in range(i + 1, n) if matrix[i][j])
    identifier = f"{family}-V{n}E{ms}-s{seed}"
    instance = Instance(identifier, graph_hash(n, edges), "external_test", n, edges, costs,
                        family, instance_content_hash(n, edges, costs),
                        {"hash_algorithm": "sha256:instance-v1", "source_member": member,
                         "nominal_edges": int(ms), "source_seed": int(seed)})
    validate_instance(instance)
    return instance
