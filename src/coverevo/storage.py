"""Lossless JSON encoding and atomic local persistence helpers."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
import json
import os
from pathlib import Path
from typing import Any

from .contracts import SolveResult


_TYPE_KEY = "__coverevo_type__"


def encode_lossless(value: Any) -> Any:
    """Convert authoritative rational/decimal/tuple values to JSON-safe markers."""
    if isinstance(value, Fraction):
        return {
            _TYPE_KEY: "fraction",
            "numerator": str(value.numerator),
            "denominator": str(value.denominator),
        }
    if isinstance(value, Decimal):
        return {_TYPE_KEY: "decimal", "value": str(value)}
    if isinstance(value, tuple):
        return {_TYPE_KEY: "tuple", "items": [encode_lossless(item) for item in value]}
    if isinstance(value, list):
        return [encode_lossless(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("lossless JSON mappings require string keys")
        return {key: encode_lossless(item) for key, item in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported lossless JSON value: {type(value).__name__}")


def decode_lossless(value: Any) -> Any:
    """Restore values emitted by :func:`encode_lossless`."""
    if isinstance(value, list):
        return [decode_lossless(item) for item in value]
    if not isinstance(value, dict):
        return value
    marker = value.get(_TYPE_KEY)
    if marker == "fraction":
        return Fraction(int(value["numerator"]), int(value["denominator"]))
    if marker == "decimal":
        return Decimal(value["value"])
    if marker == "tuple":
        return tuple(decode_lossless(item) for item in value["items"])
    if marker is not None:
        raise ValueError(f"unknown lossless JSON marker: {marker}")
    return {key: decode_lossless(item) for key, item in value.items()}


def dump_json_atomic(path: str | Path, value: Any) -> None:
    """Flush a JSON document in the target directory and atomically replace it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    encoded = encode_lossless(value)
    payload = json.dumps(
        encoded, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def load_json_lossless(path: str | Path) -> Any:
    source = Path(path)
    return decode_lossless(json.loads(source.read_text(encoding="utf-8")))


def solve_result_to_mapping(result: SolveResult) -> dict[str, Any]:
    return {
        "success": result.success,
        "selected_before_pruning": result.selected_before_pruning,
        "selected": result.selected,
        "cost_before_pruning": result.cost_before_pruning,
        "raw_cost": result.raw_cost,
        "operational_cost": result.operational_cost,
        "construction_steps": result.construction_steps,
        "failure_reason": result.failure_reason,
        "runtime_ns": result.runtime_ns,
        "trace": result.trace,
    }


def solve_result_from_mapping(data: dict[str, Any]) -> SolveResult:
    return SolveResult(
        success=data["success"],
        selected_before_pruning=tuple(data["selected_before_pruning"]),
        selected=tuple(data["selected"]),
        cost_before_pruning=data["cost_before_pruning"],
        raw_cost=data["raw_cost"],
        operational_cost=data["operational_cost"],
        construction_steps=data["construction_steps"],
        failure_reason=data["failure_reason"],
        runtime_ns=data["runtime_ns"],
        trace=data["trace"],
    )
