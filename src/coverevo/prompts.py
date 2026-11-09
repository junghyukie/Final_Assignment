"""Render the frozen v1 prompt templates and validate provider JSON output."""

from __future__ import annotations

from fractions import Fraction
import json
import math
from pathlib import Path
from typing import Any

from .contracts import ExperimentConfig, Program


MAX_RESPONSE_BYTES = 16_384
MAX_RATIONALE_CHARS = 350
_PROMPT_DIRECTORY = Path(__file__).resolve().parents[2] / "prompts"


class CandidateResponseError(ValueError):
    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


def _read_prompt(name: str) -> str:
    return (_PROMPT_DIRECTORY / name).read_text(encoding="utf-8")


def _fixed_json(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (float, Fraction)):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("feedback contains a non-finite number")
        return f"{numeric:.6f}"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ",".join(_fixed_json(item) for item in value) + "]"
    if isinstance(value, tuple):
        return "[" + ",".join(_fixed_json(item) for item in value) + "]"
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("feedback JSON keys must be strings")
        return "{" + ",".join(
            json.dumps(key, ensure_ascii=False) + ":" + _fixed_json(value[key])
            for key in sorted(value)
        ) + "}"
    raise TypeError(f"unsupported feedback value: {type(value).__name__}")


def _join_task_and_instruction(instruction: str) -> str:
    task = _read_prompt("task.txt")
    return task.rstrip("\r\n") + "\n\n" + instruction.rstrip("\r\n") + "\n"


def build_messages(
    mode: str,
    parents: list[Program],
    feedback: list[dict[str, Any]] | None,
    config: ExperimentConfig,
) -> list[dict[str, str]]:
    """Build one system/user message pair with only mode-authorized context."""
    del config
    normalized = mode.upper()
    ordered_parents = sorted(parents, key=lambda program: program.program_hash)
    if normalized in {"IND", "FRESH"}:
        if parents or feedback is not None:
            raise ValueError("independent prompts cannot contain parents or feedback")
        instruction = _read_prompt("independent.txt")
    elif normalized == "EVO_P":
        if not ordered_parents:
            raise ValueError("EVO_P parent prompt requires at least one parent")
        if feedback is not None:
            raise ValueError("EVO_P must not receive numeric feedback")
        parent_json = json.dumps(
            [
                {"expression": parent.expression, "rationale": parent.rationale}
                for parent in ordered_parents
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        instruction = _read_prompt("parent_only.txt").replace(
            "{{parents_expression_and_rationale_json}}", parent_json
        )
    elif normalized == "EVO_F":
        if not ordered_parents or feedback is None:
            raise ValueError("EVO_F parent prompt requires parents and feedback")
        by_hash = {item.get("program_hash"): item for item in feedback}
        ordered_hashes = [program.program_hash for program in ordered_parents]
        if set(by_hash) != set(ordered_hashes):
            raise ValueError("feedback hashes must match parent hashes exactly")
        parent_json = json.dumps(
            [
                {"expression": parent.expression, "rationale": parent.rationale}
                for parent in ordered_parents
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        feedback_json = _fixed_json([by_hash[program_hash] for program_hash in ordered_hashes])
        instruction = (
            _read_prompt("parent_feedback.txt")
            .replace("{{parents_expression_and_rationale_json}}", parent_json)
            .replace("{{training_feedback_json}}", feedback_json)
        )
    else:
        raise ValueError(f"unknown prompt mode: {mode}")

    system = _read_prompt("system.txt")
    user = _join_task_and_instruction(instruction)
    if "{{" in system or "{{" in user:
        raise ValueError("prompt contains an unresolved placeholder")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_candidate_response(content: str | None) -> tuple[str, str]:
    """Require exactly one bounded JSON object; never repair model output."""
    if not isinstance(content, str) or len(content.encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise CandidateResponseError("INVALID_SCHEMA", "response is absent or exceeds 16 KiB")
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise CandidateResponseError("INVALID_JSON", "response is not exact JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"expression", "rationale"}:
        raise CandidateResponseError("INVALID_SCHEMA", "response must have exactly two keys")
    expression = payload["expression"]
    rationale = payload["rationale"]
    if not isinstance(expression, str) or not isinstance(rationale, str):
        raise CandidateResponseError("INVALID_SCHEMA", "expression and rationale must be strings")
    if len(rationale) > MAX_RATIONALE_CHARS:
        raise CandidateResponseError("INVALID_SCHEMA", "rationale exceeds 350 characters")
    return expression, rationale
