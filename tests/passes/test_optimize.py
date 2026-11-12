"""optimize: canonical operators, simplifications, folding, CSE and DCE preserve semantics."""

from __future__ import annotations

import numpy as np
import pytest
from graph_strategies import graphs
from hypothesis import HealthCheck, given, settings

from tessera.dtype import DType
from tessera.ir import Graph, TensorType, Value, interpret, random_inputs
from tessera.ir.opdefs import CANONICAL_OPS
from tessera.passes import optimize

F32 = DType.float32
RNG = np.random.default_rng(7)


def const(g: Graph, *shape: int, name: str = "c") -> Value:
    return g.add_constant(RNG.standard_normal(shape).astype(np.float32), name)


def same_outputs(before: Graph, after: Graph, rtol: float = 1e-4) -> None:
    feeds = random_inputs(before, seed=1)
    want = interpret(before, feeds, float64=True)
    got = interpret(after, feeds)
    assert len(got) == len(want)
    # By position: an output that optimizes to a graph input takes the input's name.
    for out, value in zip(after.outputs, want.values(), strict=True):
        np.testing.assert_allclose(got[out.name], value, rtol=rtol, atol=1e-5)


def ops(g: Graph) -> list[str]:
    return [n.op for n in g.nodes]


def test_noops_disappear() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((2, 3), F32))
    y = g.op("Reshape", x, shape=(2, 3))
    y = g.op("Transpose", y, perm=(0, 1))
    y = g.op("Cast", y, to="float32")
    y = g.op("Slice", y, starts=(0,), ends=(2,), axes=(0,))
    y = g.op("Pad", y, pads=(0, 0, 0, 0))
    y = g.op("Add", y, g.add_constant(np.zeros(3, np.float32), "zero"))
    y = g.op("Mul", g.add_constant(np.float32(1), "one"), y)
    y = g.op("Relu", g.op("Concat", y, axis=0))
    g.set_outputs([y])
    out = optimize(g)
    assert ops(out) == ["Relu"]
    assert out.outputs[0].name == y.name
    same_outputs(g, out)


def test_broadcasting_neutral_operands_stay() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((3,), F32))
    g.set_outputs([g.op("Add", x, g.add_constant(np.zeros((2, 3), np.float32), "zero"))])
    assert ops(optimize(g)) == ["Add"]


def test_reshape_and_transpose_chains_merge() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((2, 3, 4), F32))
    r = g.op("Reshape", g.op("Reshape", x, shape=(6, 4)), shape=(4, 6))
    t = g.op("Transpose", g.op("Transpose", x, perm=(1, 0, 2)), perm=(0, 2, 1))
    back = g.op("Transpose", g.op("Transpose", x, perm=(2, 0, 1)), perm=(1, 2, 0))
    g.set_outputs([r, t, g.op("Relu", back)])
    out = optimize(g)
    assert sorted(ops(out)) == ["Relu", "Reshape", "Transpose"]
    (transpose,) = (n for n in out.nodes if n.op == "Transpose")
    assert transpose.attrs["perm"] == (1, 2, 0)
    same_outputs(g, out)


def test_transpose_of_unit_axes_becomes_reshape() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((1, 1, 1, 8), F32))
    g.set_outputs([g.op("Transpose", x, perm=(0, 3, 1, 2))])
    out = optimize(g)
    assert ops(out) == ["Reshape"]
    same_outputs(g, out)


def test_constants_fold_and_merge() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((4, 3), F32))
    w = const(g, 3, 4, name="w")
    folded = g.op("Transpose", g.op("Mul", w, g.add_constant(np.float32(2), "two")), perm=(1, 0))
    a = g.op("Add", x, g.add_constant(np.ones(3, np.float32), "one_a"))
    b = g.op("Add", x, g.add_constant(np.ones(3, np.float32), "one_b"))
    g.set_outputs([g.op("Mul", g.op("Mul", a, b), folded)])
    out = optimize(g)
    assert ops(out) == ["Add", "Mul", "Mul"]
    assert len(out.constants()) == 2
    same_outputs(g, out)
    unfolded = optimize(g, fold=False)
    assert ops(unfolded) == ["Mul", "Transpose", "Add", "Mul", "Mul"]
    same_outputs(g, unfolded)


def test_large_expansions_are_not_folded() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((1024, 1024), F32))
    big = g.op("Expand", g.add_constant(np.float32(3), "three"), shape=(1024, 1024))
    g.set_outputs([g.op("Add", x, big)])
    assert ops(optimize(g)) == ["Expand", "Add"]


def test_dead_nodes_are_removed() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((3,), F32))
    g.op("Exp", x)
    g.set_outputs([g.op("Tanh", x)])
    assert ops(optimize(g)) == ["Tanh"]


def test_batchnorm_without_a_convolution_becomes_mul_add() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((2, 3, 4), F32))
    var = g.add_constant(np.abs(RNG.standard_normal(3)).astype(np.float32) + 0.1, "var")
    g.set_outputs([g.op("BatchNorm", x, const(g, 3), const(g, 3), const(g, 3), var)])
    out = optimize(g)
    assert ops(out) == ["Mul", "Add"]
    same_outputs(g, out)


def test_unknown_layout_is_rejected() -> None:
    with pytest.raises(ValueError, match="layout"):
        optimize(Graph(), layout="NCWH")


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(graphs(import_ops=True))
def test_optimize_preserves_semantics_on_random_graphs(g: Graph) -> None:
    for fold in (True, False):
        out = optimize(g, fold=fold)
        assert {n.op for n in out.nodes} <= CANONICAL_OPS
        assert [v.name for v in out.inputs] == [v.name for v in g.inputs]
        same_outputs(g, out)
