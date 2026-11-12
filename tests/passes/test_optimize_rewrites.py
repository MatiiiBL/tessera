"""Pattern rewrites, BatchNorm folding, grouped convolutions and the NHWC layout pass."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest
from graph_strategies import graphs
from hypothesis import HealthCheck, given, settings

from tessera.dtype import DType
from tessera.ir import Graph, TensorType, Value, interpret, random_inputs
from tessera.passes import optimize

F32 = DType.float32
RNG = np.random.default_rng(11)


def const(g: Graph, *shape: int, name: str = "c", positive: bool = False) -> Value:
    data = RNG.standard_normal(shape).astype(np.float32)
    return g.add_constant(np.abs(data) + 0.5 if positive else data, name)


def scalar(g: Graph, value: float) -> Value:
    return g.add_constant(np.float32(value), "k")


def ops(g: Graph) -> Counter[str]:
    return Counter(n.op for n in g.nodes)


def same_outputs(before: Graph, after: Graph) -> None:
    feeds = random_inputs(before, seed=2)
    want = interpret(before, feeds, float64=True)
    got = interpret(after, feeds)
    for out, value in zip(after.outputs, want.values(), strict=True):
        np.testing.assert_allclose(got[out.name], value, rtol=1e-4, atol=1e-5)


def layernorm_graph(rsqrt: bool) -> Graph:
    g = Graph()
    x = g.add_input("x", TensorType((2, 3, 8), F32))
    mean = g.op("ReduceMean", x, axes=(-1,))
    d = g.op("Sub", x, mean)
    var = g.op("ReduceMean", g.op("Mul", d, d), axes=(2,))
    shifted = g.op("Add", scalar(g, 1e-6), var)
    if rsqrt:
        normed = g.op("Mul", g.op("Div", scalar(g, 1.0), g.op("Sqrt", shifted)), d)
    else:
        normed = g.op("Div", d, g.op("Sqrt", shifted))
    y = g.op("Add", const(g, 8, name="beta"), g.op("Mul", normed, const(g, 8, name="gamma")))
    g.set_outputs([y])
    return g


@pytest.mark.parametrize("rsqrt", [False, True])
def test_decomposed_layernorm_becomes_one_node(rsqrt: bool) -> None:
    g = layernorm_graph(rsqrt)
    out = optimize(g)
    assert ops(out) == Counter({"LayerNorm": 1})
    (ln,) = out.nodes
    assert ln.attrs["axis"] == 2
    assert ln.attrs["epsilon"] == pytest.approx(1e-6)
    same_outputs(g, out)
    assert ops(optimize(g, fold=False))["LayerNorm"] == 0


def test_layernorm_over_two_axes_and_scale_on_other_axes_stays() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((2, 3, 4), F32))
    mean = g.op("ReduceMean", x, axes=(1, 2))
    d = g.op("Sub", x, mean)
    var = g.op("ReduceMean", g.op("Pow", d, scalar(g, 2.0)), axes=(1, 2))
    normed = g.op("Div", d, g.op("Sqrt", g.op("Add", var, scalar(g, 1e-5))))
    g.set_outputs([g.op("Mul", normed, const(g, 2, 1, 1))])
    out = optimize(g)
    assert ops(out) == Counter({"LayerNorm": 1, "Mul": 1})
    assert out.nodes[0].attrs["axis"] == 1
    same_outputs(g, out)


def gelu_erf(g: Graph, x: Value, variant: int) -> Value:
    if variant == 0:  # x * 0.5 * (1 + erf(x / sqrt(2))), as BERT exports it
        erf = g.op("Erf", g.op("Div", x, scalar(g, 1.4142135)))
        return g.op("Mul", g.op("Mul", x, scalar(g, 0.5)), g.op("Add", erf, scalar(g, 1.0)))
    erf = g.op("Erf", g.op("Mul", scalar(g, 0.70710678), x))
    return g.op("Mul", x, g.op("Mul", g.op("Add", scalar(g, 1.0), erf), scalar(g, 0.5)))


def gelu_tanh(g: Graph, x: Value, variant: int) -> Value:
    cube = g.op("Pow", x, scalar(g, 3.0)) if variant == 0 else g.op("Mul", x, g.op("Mul", x, x))
    inner = g.op("Add", x, g.op("Mul", scalar(g, 0.044715), cube))
    t = g.op("Tanh", g.op("Mul", inner, scalar(g, math.sqrt(2 / math.pi))))
    return g.op("Mul", g.op("Mul", scalar(g, 0.5), x), g.op("Add", scalar(g, 1.0), t))


@pytest.mark.parametrize("variant", [0, 1])
@pytest.mark.parametrize(("build", "approximate"), [(gelu_erf, "none"), (gelu_tanh, "tanh")])
def test_gelu_forms(build: object, approximate: str, variant: int) -> None:
    g = Graph()
    x = g.add_input("x", TensorType((4, 16), F32))
    g.set_outputs([build(g, x, variant)])  # type: ignore[operator]
    out = optimize(g)
    assert ops(out) == Counter({"Gelu": 1})
    assert out.nodes[0].attrs["approximate"] == approximate
    same_outputs(g, out)


def test_gelu_with_a_wrong_constant_stays_decomposed() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((4, 16), F32))
    erf = g.op("Erf", g.op("Div", x, scalar(g, 1.5)))
    g.set_outputs([g.op("Mul", g.op("Mul", x, scalar(g, 0.5)), g.op("Add", erf, scalar(g, 1.0)))])
    assert ops(optimize(g))["Gelu"] == 0


def test_silu_hardswish_and_rsqrt() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((3, 5), F32))
    silu = g.op("Mul", g.op("Sigmoid", x), x)
    hard = g.op("Mul", x, g.op("HardSigmoid", x, alpha=1 / 6, beta=0.5))
    root = g.op("Div", scalar(g, 1.0), g.op("Sqrt", g.op("Abs", x)))
    g.set_outputs([silu, hard, root])
    out = optimize(g)
    assert ops(out) == Counter({"Silu": 1, "HardSwish": 1, "Abs": 1, "Rsqrt": 1})
    same_outputs(g, out)


def conv_bn(g: Graph, x: Value, channels: int, out: int, bias: bool) -> Value:
    args = [x, const(g, out, channels, 3, 3)] + ([const(g, out)] if bias else [])
    y = g.op("Conv2d", *args, pads=(1, 1))
    stats = [const(g, out), const(g, out), const(g, out), const(g, out, positive=True)]
    return g.op("BatchNorm", y, *stats)


@pytest.mark.parametrize("bias", [False, True])
def test_batchnorm_folds_into_the_convolution(bias: bool) -> None:
    g = Graph()
    x = g.add_input("x", TensorType((1, 3, 6, 6), F32))
    g.set_outputs([g.op("Relu", conv_bn(g, x, 3, 4, bias))])
    out = optimize(g, layout="NCHW")
    assert ops(out) == Counter({"Conv2d": 1, "Relu": 1})
    same_outputs(g, out)
    unfolded = optimize(g, layout="NCHW", fold=False)
    assert ops(unfolded) == Counter({"Conv2d": 1, "Mul": 1, "Add": 1, "Relu": 1})
    same_outputs(g, unfolded)


def test_batchnorm_after_a_shared_convolution_is_decomposed() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((1, 2, 4, 4), F32))
    y = g.op("Conv2d", x, const(g, 2, 2, 1, 1))
    stats = [const(g, 2), const(g, 2), const(g, 2), const(g, 2, positive=True)]
    g.set_outputs([g.op("BatchNorm", y, *stats), y])
    out = optimize(g, layout="NCHW")
    assert ops(out) == Counter({"Conv2d": 1, "Mul": 1, "Add": 1})
    same_outputs(g, out)


@pytest.mark.parametrize("layout", ["NCHW", "NHWC"])
def test_grouped_convolution_is_split(layout: str) -> None:
    g = Graph()
    x = g.add_input("x", TensorType((1, 6, 5, 5), F32))
    g.set_outputs([g.op("Conv2d", x, const(g, 4, 3, 3, 3), const(g, 4), groups=2, pads=(1, 1))])
    out = optimize(g, layout=layout)
    assert ops(out)["Conv2d"] == 2
    assert all(n.attrs["groups"] == 1 for n in out.nodes if n.op == "Conv2d")
    same_outputs(g, out)


def small_cnn() -> Graph:
    g = Graph("cnn")
    x = g.add_input("x", TensorType((1, 3, 12, 12), F32))
    y = g.op("Relu", conv_bn(g, x, 3, 8, False))
    y = g.op("MaxPool2d", y, kernel=2, strides=2)
    skip = y
    y = g.op("Clip", g.op("Conv2d", y, const(g, 8, 1, 3, 3), groups=8, pads=(1, 1)), min=0, max=6)
    y = g.op("Conv2d", y, const(g, 8, 8, 1, 1), const(g, 8))
    y = g.op("Add", y, skip)
    y = g.op("Mul", y, const(g, 8, 1, 1))
    y = g.op("Concat", y, g.op("AvgPool2d", y, kernel=3, pads=(1, 1), count_include_pad=1), axis=1)
    y = g.op("Pad", g.op("Softmax", y, axis=1), pads=(0, 0, 1, 0, 0, 0, 0, 1))
    pooled = g.op("ReduceMean", y, axes=(2, 3))
    flat = g.op("Reshape", pooled, shape=(1, -1))
    g.set_outputs([g.op("MatMul", flat, const(g, 16, 5)), y])
    return g


def test_cnn_has_transposes_only_at_inputs_and_outputs() -> None:
    g = small_cnn()
    for fold in (True, False):
        out = optimize(g, fold=fold)
        assert all(n.attrs["layout"] == "NHWC" for n in out.nodes if "layout" in n.attrs)
        transposes = [n for n in out.nodes if n.op == "Transpose"]
        assert len(transposes) == 2, out
        x = out.inputs[0]
        assert {n.inputs[0].name for n in transposes} >= {x.name}
        assert any(n.output in out.outputs for n in transposes)
        same_outputs(g, out)
    for conv in (n for n in optimize(g).nodes if n.op == "Conv2d"):
        assert conv.inputs[1].shape[3] == conv.output.shape[3]  # HWIO weights
        assert conv.inputs[1].is_constant


def test_nchw_layout_keeps_the_graph_in_nchw() -> None:
    out = optimize(small_cnn(), layout="NCHW")
    assert "Transpose" not in ops(out)
    assert all(n.attrs["layout"] == "NCHW" for n in out.nodes if "layout" in n.attrs)


def test_reductions_without_keepdims_and_layernorm_follow_the_layout() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((1, 4, 5, 5), F32))
    y = g.op("Conv2d", x, const(g, 4, 4, 1, 1))
    reduced = g.op("ReduceSum", y, axes=(1,), keepdims=0)
    sliced = g.op("Slice", y, starts=(1, 0), ends=(3, 4), axes=(1, 3))
    normed = g.op("LayerNorm", g.op("Relu", y), const(g, 5), const(g, 5))
    g.set_outputs([reduced, sliced, normed])
    out = optimize(g)
    same_outputs(g, out)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(graphs(import_ops=True))
def test_layouts_preserve_semantics_on_random_graphs(g: Graph) -> None:
    for layout in ("NHWC", "NCHW"):
        out = optimize(g, layout=layout)
        if layout == "NHWC":
            assert all(n.attrs.get("layout", "NHWC") == "NHWC" for n in out.nodes)
        same_outputs(g, out)
