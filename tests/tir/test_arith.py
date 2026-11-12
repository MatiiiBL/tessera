"""The simplifier and the interval analyses agree with evaluating the expressions."""

from __future__ import annotations

import itertools

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tessera.dtype import DType
from tessera.tir.arith import Analyzer, Interval, simplify
from tessera.tir.expr import (
    Binary,
    Compare,
    Expr,
    IntImm,
    Logical,
    Var,
    ge,
    gt,
    le,
    logical_and,
    lt,
    maximum,
    minimum,
)
from tessera.tir.printer import script

VI, VJ, VK = Var("i"), Var("j"), Var("k")
RANGES = {VI: Interval(0, 3), VJ: Interval(0, 3), VK: Interval(0, 11)}


def evaluate(e: Expr, env: dict[Var, int]) -> int:
    if isinstance(e, IntImm):
        return e.value
    if isinstance(e, Var):
        return env[e]
    if isinstance(e, Binary):
        a, b = evaluate(e.a, env), evaluate(e.b, env)
        return {
            "add": lambda: a + b,
            "sub": lambda: a - b,
            "mul": lambda: a * b,
            "floordiv": lambda: a // b,
            "floormod": lambda: a % b,
            "min": lambda: min(a, b),
            "max": lambda: max(a, b),
        }[e.op]()
    if isinstance(e, Compare):
        a, b = evaluate(e.a, env), evaluate(e.b, env)
        return int(
            {"lt": a < b, "le": a <= b, "eq": a == b, "ne": a != b, "gt": a > b, "ge": a >= b}[e.op]
        )
    if isinstance(e, Logical):
        a, b = evaluate(e.a, env), evaluate(e.b, env)
        return int(bool(a) and bool(b)) if e.op == "and" else int(bool(a) or bool(b))
    raise TypeError(e)


def points() -> list[dict[Var, int]]:
    ranges = [range(r.lo or 0, (r.hi or 0) + 1) for r in RANGES.values()]
    return [dict(zip(RANGES, p, strict=True)) for p in itertools.product(*ranges)]


def exprs() -> st.SearchStrategy[Expr]:
    leaves = st.one_of(st.sampled_from([VI, VJ, VK]), st.integers(-6, 9).map(IntImm))

    def extend(children: st.SearchStrategy[Expr]) -> st.SearchStrategy[Expr]:
        arith = st.tuples(st.sampled_from(["add", "sub", "min", "max"]), children, children).map(
            lambda t: Binary(t[0], t[1], t[2])  # type: ignore[arg-type]
        )
        scaled = st.tuples(children, st.integers(-3, 5)).map(lambda t: t[0] * t[1])
        div = st.tuples(st.sampled_from(["floordiv", "floormod"]), children, st.integers(1, 6)).map(
            lambda t: Binary(t[0], t[1], IntImm(t[2]))  # type: ignore[arg-type]
        )
        return st.one_of(arith, scaled, div)

    return st.recursive(leaves, extend, max_leaves=8)


@settings(max_examples=300, deadline=None)
@given(exprs())
def test_simplify_preserves_values(e: Expr) -> None:
    s = simplify(e, RANGES)
    for env in points():
        assert evaluate(s, env) == evaluate(e, env), f"{script(e)} -> {script(s)}"


@settings(max_examples=300, deadline=None)
@given(exprs(), st.sampled_from(["lt", "le", "eq", "ne", "gt", "ge"]), exprs())
def test_simplify_preserves_comparisons(a: Expr, op: str, b: Expr) -> None:
    e = Compare(op, a, b)  # type: ignore[arg-type]
    s = simplify(e, RANGES)
    for env in points():
        assert evaluate(s, env) == evaluate(e, env), f"{script(e)} -> {script(s)}"


@settings(max_examples=300, deadline=None)
@given(exprs())
def test_bound_contains_every_value(e: Expr) -> None:
    b = Analyzer(RANGES).bound(e)
    for env in points():
        v = evaluate(e, env)
        assert b.lo is None or b.lo <= v
        assert b.hi is None or v <= b.hi


@settings(max_examples=300, deadline=None)
@given(exprs())
def test_relax_bounds_hold_for_every_outer_value(e: Expr) -> None:
    an = Analyzer(RANGES)
    lo, hi = an.relax(e, {VJ: RANGES[VJ], VK: RANGES[VK]})
    for env in points():
        v = evaluate(e, env)
        if lo is not None:
            assert evaluate(lo, env) <= v
        if hi is not None:
            assert v <= evaluate(hi, env)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ((VI * 4 + VJ) - VI * 4, "j"),
        ((VI * 8 + VJ) // 4, "i * 2"),
        ((VI * 4 + VJ) // 4, "i"),
        ((VI * 4 + VJ) % 4, "j"),
        ((VK // 4) * 4 + VK % 4, "k"),
        (VK - (VK // 4) * 4, "k % 4"),
        ((VK // 4) // 2, "k // 8"),
        ((VK // 4) // 3, "0"),
        ((VK % 8) % 4, "k % 4"),
        (minimum(VI, VI + 1), "i"),
        (maximum(VI, VJ), "max(i, j)"),
        (lt(VI * 4 + VJ, 16), "True"),
        (lt(VI * 4 + VJ, 10), "i * 4 + j < 10"),
        (ge(VI * 4 + VJ, 0), "True"),
        (lt(VI * 4, 8), "i < 2"),
        (le(VI, 2), "i < 3"),
        (gt(VI, 1), "i >= 2"),
        (logical_and(lt(VI * 4 + VJ, 10), lt(VI * 4 + VJ, 12)), "i * 4 + j < 10"),
    ],
)
def test_simplification_rules(expr: Expr, expected: str) -> None:
    assert script(simplify(expr, RANGES)) == expected


def test_float_expressions_fold_but_never_reassociate() -> None:
    x = Var("x", DType.float32)
    from tessera.tir.expr import FloatImm

    assert script(simplify(FloatImm(1.5) * FloatImm(2.0) + x * 1.0)) == "3.0 + x"
    assert script(simplify((x + 1.0) + 2.0)) == "x + 1.0 + 2.0"
    third = simplify(FloatImm(1.0) / FloatImm(3.0))
    assert isinstance(third, FloatImm)
    assert third.value == pytest.approx(1 / 3, rel=1e-07)


def test_relax_gives_tile_bounds() -> None:
    an = Analyzer(RANGES)
    lo, hi = an.relax(VI * 4 + VJ, {VJ: Interval(0, 3)})
    assert lo is not None
    assert hi is not None
    assert (script(lo), script(hi)) == ("i * 4", "i * 4 + 3")
