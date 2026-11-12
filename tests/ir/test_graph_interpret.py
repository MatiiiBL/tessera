"""The graph interpreter and the graph helpers it relies on."""

from __future__ import annotations

import numpy as np
import pytest

from tessera.dtype import DType
from tessera.ir import Graph, TensorType, interpret, random_inputs

F32 = DType.float32


def mlp() -> Graph:
    rng = np.random.default_rng(0)
    g = Graph("mlp")
    x = g.add_input("x", TensorType((2, 4), F32))
    w1 = g.add_constant(rng.standard_normal((4, 8)).astype(np.float32), "w1")
    b1 = g.add_constant(rng.standard_normal(8).astype(np.float32), "b1")
    w2 = g.add_constant(rng.standard_normal((8, 3)).astype(np.float32), "w2")
    h = g.op("Relu", g.op("Add", g.op("MatMul", x, w1), b1), name="hidden")
    y = g.op("Softmax", g.op("MatMul", h, w2), name="probs")
    g.set_outputs([y])
    return g


def test_interpret_matches_numpy() -> None:
    g = mlp()
    x = np.random.default_rng(1).standard_normal((2, 4)).astype(np.float32)
    consts = {v.name: v.data for v in g.constants()}
    h = np.maximum(x @ consts["w1"] + consts["b1"], 0)
    logits = h @ consts["w2"]
    e = np.exp(logits - logits.max(-1, keepdims=True))
    out = interpret(g, {"x": x})
    assert list(out) == ["probs"]
    np.testing.assert_allclose(out["probs"], e / e.sum(-1, keepdims=True), rtol=1e-6)


def test_float64_mode_computes_in_float64() -> None:
    g = mlp()
    x = random_inputs(g, seed=3)
    lo = interpret(g, x)["probs"]
    hi = interpret(g, x, float64=True)["probs"]
    assert lo.dtype == np.float32
    assert hi.dtype == np.float64
    np.testing.assert_allclose(lo, hi, rtol=1e-5)


def test_float64_mode_still_rounds_at_casts() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((1,), F32))
    y = g.op("Cast", g.op("Cast", x, to="float16"), to="float32")
    g.set_outputs([y])
    out = interpret(g, {"x": np.array([1 / 3], np.float32)}, float64=True)
    assert out[y.name].dtype == np.float64
    assert out[y.name][0] == np.float64(np.float16(1 / 3))


def test_intermediates_returns_every_value() -> None:
    g = mlp()
    out = interpret(g, random_inputs(g), intermediates=True)
    assert set(out) == {n.output.name for n in g.nodes}


def test_outputs_are_copies() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((3,), F32))
    c = g.add_constant(np.ones(3, np.float32), "c")
    g.set_outputs([x, c, g.op("Reshape", c, shape=(1, 3))])
    data = np.zeros(3, np.float32)
    out = interpret(g, {"x": data})
    for array in out.values():
        array += 5
    assert data[0] == 0
    assert c.data is not None
    assert c.data[0] == 1


@pytest.mark.parametrize(
    ("feeds", "error", "match"),
    [
        ({}, KeyError, "missing input 'x'"),
        ({"x": np.zeros((2, 4), np.float32), "z": np.zeros(1)}, KeyError, "no inputs"),
        ({"x": np.zeros((4, 2), np.float32)}, ValueError, "shape"),
        ({"x": np.zeros((2, 4), np.float64)}, ValueError, "type"),
    ],
)
def test_bad_inputs(feeds: dict[str, np.ndarray], error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        interpret(mlp(), feeds)


def test_failing_reference_names_the_node() -> None:
    g = Graph()
    data = g.add_input("data", TensorType((3,), F32))
    idx = g.add_input("idx", TensorType((1,), DType.int64))
    g.set_outputs([g.op("Gather", data, idx, name="lookup")])
    with pytest.raises(IndexError) as info:
        interpret(g, {"data": np.zeros(3, np.float32), "idx": np.array([7])})
    assert any("lookup" in note for note in info.value.__notes__)


def test_random_inputs_cover_every_dtype() -> None:
    g = Graph()
    g.add_input("f", TensorType((2, 2), F32))
    g.add_input("i", TensorType((5,), DType.int64))
    g.add_input("b", TensorType((3,), DType.bool_))
    feeds = random_inputs(g, int_high=4)
    assert feeds["f"].dtype == np.float32
    assert feeds["i"].dtype == np.int64
    assert feeds["i"].max() < 4
    assert feeds["b"].dtype == np.bool_


def test_topo_sort_moves_late_producers_before_their_first_reader() -> None:
    g = mlp()
    late = g.nodes.pop(0)
    g.nodes.append(late)
    g.topo_sort()
    g.verify()
    assert g.nodes[0] is late
    g.remove_nodes([late])
    with pytest.raises(ValueError, match="no node defines"):
        g.topo_sort()


def test_name_outputs_renames_or_wraps_in_a_view() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((3,), F32))
    c = g.add_constant(np.ones(3, np.float32), "c")
    r = g.op("Relu", x, name="r")
    g.set_outputs([r, r, x, c, g.op("Neg", r, name="taken")])
    g.name_outputs(["first", "second", "third", "fourth", "r"])
    g.verify()
    assert [v.name for v in g.outputs] == ["first", "second", "third", "fourth", "r"]
    assert g.outputs[0] is r
    assert [n.op for n in g.nodes] == ["Relu", "Neg", "Reshape", "Reshape", "Reshape"]
    out = interpret(g, {"x": np.array([-1.0, 0.0, 2.0], np.float32)})
    np.testing.assert_array_equal(out["second"], [0.0, 0.0, 2.0])
    np.testing.assert_array_equal(out["r"], [0.0, 0.0, -2.0])
    with pytest.raises(ValueError, match="names for"):
        g.name_outputs(["only_one"])


def test_name_outputs_never_takes_an_input_name() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((3,), F32))
    g.set_outputs([g.op("Relu", x)])
    with pytest.raises(ValueError, match="name of a graph input"):
        g.name_outputs(["x"])
