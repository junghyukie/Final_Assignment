"""Independent brute-force oracle and status-safe OR-Tools CP-SAT adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import version
import contextlib
import io
import math
from typing import Any, Protocol

from .contracts import Instance, OracleResult
from .instances import validate_instance
from .solver import check_solution, greedy_reference


class OracleValidationError(RuntimeError):
    """Raised when backend output contradicts the always-feasible MWDS model."""


@dataclass(frozen=True, slots=True)
class CpSatRawResult:
    status: str
    selected: tuple[int, ...] | None
    objective: float | None
    best_bound: float | None
    wall_time_s: float
    metadata: dict[str, Any] = field(default_factory=dict)


class ExactBackend(Protocol):
    def solve(self, instance: Instance, time_limit_s: float) -> CpSatRawResult: ...


def brute_force_optimum(instance: Instance) -> tuple[int, tuple[int, ...]]:
    """Enumerate all subsets for independently checked graphs with ``n <= 14``."""
    validate_instance(instance)
    if instance.n > 14:
        raise ValueError("brute_force_optimum requires n <= 14")

    best_cost: int | None = None
    best_selected: tuple[int, ...] | None = None
    for mask in range(1, 1 << instance.n):
        selected = tuple(
            node for node in range(instance.n) if mask & (1 << node)
        )
        cost = sum(instance.costs[node] for node in selected)
        if best_cost is not None and cost > best_cost:
            continue
        if not check_solution(instance, selected):
            continue
        if (
            best_cost is None
            or cost < best_cost
            or (cost == best_cost and selected < best_selected)
        ):
            best_cost = cost
            best_selected = selected

    if best_cost is None or best_selected is None:
        raise OracleValidationError("MWDS brute force found no solution although V is feasible")
    return best_cost, best_selected


class OrToolsCpSatBackend:
    """Thin adapter around the locally installed OR-Tools CP-SAT API."""

    def solve(self, instance: Instance, time_limit_s: float) -> CpSatRawResult:
        # OR-Tools 9.12 on Windows prints each DLL path from its package loader.
        # Suppress that stdout-only dependency noise; import exceptions/stderr remain visible.
        with contextlib.redirect_stdout(io.StringIO()):
            from ortools.sat.python import cp_model

        model = cp_model.CpModel()
        variables = [model.new_bool_var(f"x_{node}") for node in range(instance.n)]
        neighborhoods = [set((node,)) for node in range(instance.n)]
        for u, v in instance.edges:
            neighborhoods[u].add(v)
            neighborhoods[v].add(u)
        for neighborhood in neighborhoods:
            model.add(sum(variables[node] for node in neighborhood) >= 1)
        model.minimize(
            sum(instance.costs[node] * variables[node] for node in range(instance.n))
        )

        hint = set(greedy_reference(instance).selected)
        for node, variable in enumerate(variables):
            model.add_hint(variable, int(node in hint))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_s
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        status_code = solver.solve(model)
        status = solver.status_name(status_code).upper()
        selected: tuple[int, ...] | None = None
        objective: float | None = None
        if status in {"OPTIMAL", "FEASIBLE"}:
            selected = tuple(
                node for node, variable in enumerate(variables) if solver.value(variable)
            )
            objective = float(solver.objective_value)
        best_bound = (
            float(solver.best_objective_bound)
            if status not in {"MODEL_INVALID"}
            else None
        )
        return CpSatRawResult(
            status=status,
            selected=selected,
            objective=objective,
            best_bound=best_bound,
            wall_time_s=float(solver.wall_time),
            metadata={
                "backend": "OR_TOOLS_CP_SAT",
                "ortools_version": version("ortools"),
                "workers": 1,
                "random_seed": 0,
                "time_limit_s": time_limit_s,
                "hint": "GR+P",
            },
        )


def _interpret_result(instance: Instance, raw: CpSatRawResult) -> OracleResult:
    status = raw.status.upper()
    recognized = {"OPTIMAL", "FEASIBLE", "UNKNOWN", "MODEL_INVALID", "INFEASIBLE"}
    if status not in recognized:
        raise OracleValidationError(f"unknown CP-SAT status: {raw.status}")
    if status == "INFEASIBLE":
        raise OracleValidationError("MWDS is always feasible by selecting every vertex")

    selected: tuple[int, ...] | None = None
    incumbent_cost: int | None = None
    if status in {"OPTIMAL", "FEASIBLE"}:
        if raw.selected is None or raw.objective is None:
            raise OracleValidationError("solution status is missing an incumbent")
        selected = tuple(sorted(raw.selected))
        if not check_solution(instance, selected):
            raise OracleValidationError("backend returned an infeasible incumbent")
        incumbent_cost = sum(instance.costs[node] for node in selected)
        if not math.isclose(raw.objective, incumbent_cost, rel_tol=0.0, abs_tol=1e-6):
            raise OracleValidationError("backend objective does not match selected-set cost")
        if raw.best_bound is not None and raw.best_bound > incumbent_cost + 1e-6:
            raise OracleValidationError("backend lower bound exceeds incumbent cost")
    elif raw.selected is not None or raw.objective is not None:
        raise OracleValidationError("non-solution status unexpectedly contains an incumbent")

    return OracleResult(
        status=status,
        selected=selected,
        incumbent_cost=incumbent_cost,
        lower_bound=raw.best_bound,
        certified_optimal=status == "OPTIMAL",
        wall_time_s=raw.wall_time_s,
        solver_metadata=dict(raw.metadata),
    )


def solve_exact(
    instance: Instance,
    time_limit_s: float = 20.0,
    *,
    backend: ExactBackend | None = None,
) -> OracleResult:
    """Solve MWDS through an injected or default CP-SAT backend and validate it."""
    validate_instance(instance)
    if not math.isfinite(time_limit_s) or time_limit_s <= 0:
        raise ValueError("time_limit_s must be a positive finite number")
    active_backend: ExactBackend = backend if backend is not None else OrToolsCpSatBackend()
    raw = active_backend.solve(instance, time_limit_s)
    return _interpret_result(instance, raw)
