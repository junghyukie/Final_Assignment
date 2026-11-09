"""Bounded parser and exact interpreter for heuristic scoring expressions."""

from __future__ import annotations

import ast
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import re
from typing import Final

from .contracts import FeatureVector, Program


class DSLValidationError(ValueError):
    """Raised when expression syntax or metadata violates the DSL contract."""


class NumericLimitError(ArithmeticError):
    """Raised when exact evaluation exceeds a configured numeric guard."""


FEATURE_NAMES: Final = frozenset(
    {"gain", "cost", "rarity", "overlap", "remaining"}
)
FUNCTION_ARITY: Final = {"div": 2, "min": 2, "max": 2, "abs": 1}
PROVENANCE_VALUES: Final = frozenset(
    {"handwritten_seed", "handwritten", "llm", "parameter_search"}
)
MAX_EXPRESSION_CHARS: Final = 512
MAX_LOGICAL_NODES: Final = 64
MAX_DEPTH: Final = 8
MAX_CONSTANT: Final = Decimal("5")
MAX_DECIMAL_PLACES: Final = 3
DIVISION_EPSILON: Final = Fraction(1, 1_000_000)
MAX_INTERMEDIATE: Final = Fraction(1_000_000)
RATIONAL_BIT_LIMIT: Final = 2048
_DECIMAL_LITERAL = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]{1,3})?\Z")


def _constant_decimal(node: ast.Constant, source: str) -> Decimal:
    if type(node.value) not in (int, float):
        raise DSLValidationError("constants must be ordinary decimal numbers")
    lexeme = ast.get_source_segment(source, node)
    if lexeme is None or _DECIMAL_LITERAL.fullmatch(lexeme) is None:
        raise DSLValidationError(
            "constants require ordinary notation with at most 3 decimal places"
        )
    try:
        value = Decimal(lexeme)
    except InvalidOperation as error:
        raise DSLValidationError("invalid decimal constant") from error
    if value < 0 or value > MAX_CONSTANT:
        raise DSLValidationError("constant absolute value must not exceed 5")
    decimal_places = max(0, -value.as_tuple().exponent)
    if decimal_places > MAX_DECIMAL_PLACES:
        raise DSLValidationError("constant has more than 3 decimal places")
    return value


def _canonical_decimal(value: Decimal) -> str:
    if value == value.to_integral():
        return str(value.to_integral())
    return format(value.normalize(), "f")


def _analyze(node: ast.AST, source: str) -> tuple[int, int, str]:
    if isinstance(node, ast.Name):
        if node.id not in FEATURE_NAMES:
            raise DSLValidationError(f"unknown feature name: {node.id}")
        return 1, 1, node.id

    if isinstance(node, ast.Constant):
        value = _constant_decimal(node, source)
        return 1, 1, _canonical_decimal(value)

    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, ast.USub):
            raise DSLValidationError("only unary minus is allowed")
        child_nodes, child_depth, child = _analyze(node.operand, source)
        return 1 + child_nodes, 1 + child_depth, f"(-{child})"

    if isinstance(node, ast.BinOp):
        symbols = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*"}
        symbol = next(
            (value for kind, value in symbols.items() if isinstance(node.op, kind)),
            None,
        )
        if symbol is None:
            raise DSLValidationError("only addition, subtraction and multiplication are allowed")
        left_nodes, left_depth, left = _analyze(node.left, source)
        right_nodes, right_depth, right = _analyze(node.right, source)
        return (
            1 + left_nodes + right_nodes,
            1 + max(left_depth, right_depth),
            f"({left}{symbol}{right})",
        )

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTION_ARITY:
            raise DSLValidationError("function is not in the DSL whitelist")
        if node.keywords:
            raise DSLValidationError("keyword arguments are not allowed")
        expected_arity = FUNCTION_ARITY[node.func.id]
        if len(node.args) != expected_arity:
            raise DSLValidationError(
                f"{node.func.id} requires exactly {expected_arity} arguments"
            )
        analyzed = [_analyze(argument, source) for argument in node.args]
        nodes = 1 + sum(item[0] for item in analyzed)
        depth = 1 + max(item[1] for item in analyzed)
        canonical = f"{node.func.id}({','.join(item[2] for item in analyzed)})"
        return nodes, depth, canonical

    raise DSLValidationError(f"forbidden syntax node: {type(node).__name__}")


def _parse_and_analyze(expression: str) -> tuple[ast.AST, int, int, str, str]:
    if not isinstance(expression, str):
        raise DSLValidationError("expression must be a string")
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise DSLValidationError("expression exceeds 512 characters")
    source = expression.strip()
    if not source:
        raise DSLValidationError("expression must not be empty")
    try:
        parsed = ast.parse(source, mode="eval")
    except (SyntaxError, ValueError, RecursionError) as error:
        raise DSLValidationError("expression is not valid DSL syntax") from error
    logical_nodes, depth, canonical = _analyze(parsed.body, source)
    if logical_nodes > MAX_LOGICAL_NODES:
        raise DSLValidationError("expression exceeds 64 logical AST nodes")
    if depth > MAX_DEPTH:
        raise DSLValidationError("expression exceeds maximum depth 8")
    return parsed.body, logical_nodes, depth, canonical, source


def parse_program(expression: str, provenance: str) -> Program:
    """Validate and canonicalize one bounded expression."""
    if provenance not in PROVENANCE_VALUES:
        raise DSLValidationError("unknown program provenance")
    _, logical_nodes, depth, canonical, _ = _parse_and_analyze(expression)
    program_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return Program(
        expression=expression,
        canonical_expression=canonical,
        program_hash=program_hash,
        ast_nodes=logical_nodes,
        ast_depth=depth,
        provenance=provenance,
    )


def _guard(value: Fraction) -> Fraction:
    if abs(value) > MAX_INTERMEDIATE:
        raise NumericLimitError("intermediate magnitude exceeds 1000000")
    if (
        abs(value.numerator).bit_length() > RATIONAL_BIT_LIMIT
        or value.denominator.bit_length() > RATIONAL_BIT_LIMIT
    ):
        raise NumericLimitError("rational numerator or denominator exceeds 2048 bits")
    return value


def _fraction_constant(node: ast.Constant, source: str) -> Fraction:
    return Fraction(_constant_decimal(node, source))


def _evaluate(node: ast.AST, source: str, features: FeatureVector) -> Fraction:
    if isinstance(node, ast.Name):
        return _guard(getattr(features, node.id))
    if isinstance(node, ast.Constant):
        return _guard(_fraction_constant(node, source))
    if isinstance(node, ast.UnaryOp):
        return _guard(-_evaluate(node.operand, source, features))
    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left, source, features)
        right = _evaluate(node.right, source, features)
        if isinstance(node.op, ast.Add):
            return _guard(left + right)
        if isinstance(node.op, ast.Sub):
            return _guard(left - right)
        if isinstance(node.op, ast.Mult):
            return _guard(left * right)
        raise DSLValidationError("validated program contains a forbidden binary operator")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        values = [_evaluate(argument, source, features) for argument in node.args]
        if node.func.id == "abs":
            return _guard(abs(values[0]))
        if node.func.id == "min":
            return _guard(min(values))
        if node.func.id == "max":
            return _guard(max(values))
        if node.func.id == "div":
            numerator, denominator = values
            protected = denominator
            if abs(denominator) < DIVISION_EPSILON:
                protected = DIVISION_EPSILON if denominator >= 0 else -DIVISION_EPSILON
            return _guard(numerator / protected)
    raise DSLValidationError("validated program contains an unsupported node")


def score_program(program: Program, features: FeatureVector) -> Fraction:
    """Evaluate a validated program using guarded exact rational arithmetic."""
    expected_hash = hashlib.sha256(program.canonical_expression.encode("utf-8")).hexdigest()
    if program.program_hash != expected_hash:
        raise DSLValidationError("program hash does not match canonical expression")
    root, nodes, depth, canonical, source = _parse_and_analyze(
        program.canonical_expression
    )
    if (
        canonical != program.canonical_expression
        or nodes != program.ast_nodes
        or depth != program.ast_depth
    ):
        raise DSLValidationError("program metadata does not match canonical expression")
    return _evaluate(root, source, features)
