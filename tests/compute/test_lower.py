"""Every operator's compute definition, lowered and interpreted, matches its reference."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tessera import ir, tir
from tessera.compute import ROLE, lower_group
from tessera.dtype import DType
from tessera.ir import Graph, TensorType
from tessera.plan import LoweredGroup
from tessera.tir.schedule import Schedule
from tessera.tir.stmt import Block, PrimFunc

from .harness import check_graph, check_group, single_group, values_of

F32 = DType.float32


def graph_with(
    shapes: list[tuple[int, ...]], dtypes: list[DType] | None = None
) -> tuple[Graph, list[ir.Value]]:
    g = Graph("g")
    dts = dtypes or [F32] * len(shapes)
    xs = [
        g.add_input(f"x{i}", TensorType(s, d))
        for i, (s, d) in enumerate(zip(shapes, dts, strict=True))
    ]
    return g, xs


def run_single(g: Graph, out: ir.Value, inputs: dict[str, np.ndarray] | None = None) -> None:
    g.set_outputs([out])
    inputs = inputs if inputs is not None else ir.random_inputs(g, seed=3)
    values = values_of(g, inputs)
    assert out.producer is not None
    check_group(single_group(out.producer), values)


# -- elementwise and broadcast ----------------------------------------------------------------

UNARY = [
    ("Relu", {}),
    ("Sigmoid", {}),
    ("Tanh", {}),
    ("Exp", {}),
    ("Log", {}),
    ("Sqrt", {}),
    ("Rsqrt", {}),
    ("Erf", {}),
    ("Neg", {}),
    ("Abs", {}),
    ("Silu", {}),
    ("HardSwish", {}),
    ("HardSigmoid", {"alpha": 0.3, "beta": 0.4}),
    ("Gelu", {"approximate": "none"}),
    ("Gelu", {"approximate": "tanh"}),
    ("Clip", {"min": -0.5, "max": 0.7}),
    ("Clip", {"max": 0.2}),
]


@pytest.mark.parametrize(("op", "attrs"), UNARY)
def test_unary(op: str, attrs: dict[str, object]) -> None:
    g, (x,) = graph_with([(2, 3, 4)])
    data = np.random.default_rng(0).standard_normal((2, 3, 4)).astype(np.float32)
    if op in ("Log", "Sqrt", "Rsqrt"):
        data = np.abs(data) + 0.1
    run_single(g, g.op(op, x, **attrs), {"x0": data})  # type: ignore[arg-type]


@pytest.mark.parametrize("op", ["Relu", "Neg", "Abs", "Clip"])
def test_unary_on_integers(op: str) -> None:
    g, (x,) = graph_with([(5,)], [DType.int32])
    data = np.array([-7, -1, 0, 3, 9], np.int32)
    attrs = {"min": -2.5, "max": 4.2} if op == "Clip" else {}
    run_single(g, g.op(op, x, **attrs), {"x0": data})  # type: ignore[arg-type]


@pytest.mark.parametrize("to", ["float32", "int32", "int64", "uint8", "bool"])
def test_cast(to: str) -> None:
    g, (x,) = graph_with([(6,)])
    data = np.array([-2.7, -0.0, 0.0, 1.5, 3.9, 200.0], np.float32)
    run_single(g, g.op("Cast", x, to=to), {"x0": data})


BINARY = ["Add", "Sub", "Mul", "Div", "Pow", "Maximum", "Minimum", "Equal", "Less", "Greater"]


@pytest.mark.parametrize("op", BINARY)
def test_broadcast_binary(op: str) -> None:
    g, (a, b) = graph_with([(3, 1, 4), (2, 1)])
    rng = np.random.default_rng(1)
    data = {"x0": rng.standard_normal((3, 1, 4)).astype(np.float32),
            "x1": rng.standard_normal((2, 1)).astype(np.float32)}  # fmt: skip
    if op == "Pow":
        data["x0"] = np.abs(data["x0"]) + 0.5
    run_single(g, g.op(op, a, b), data)


def test_integer_division_truncates() -> None:
    g, (a, b) = graph_with([(4,), (4,)], [DType.int64, DType.int64])
    data = {"x0": np.array([-7, 7, -8, 9]), "x1": np.array([2, -2, 3, 4])}
    run_single(g, g.op("Div", a, b), data)


def test_where_and_batchnorm() -> None:
    g, (cond, a, b) = graph_with([(2, 3), (3,), (2, 1)], [DType.bool_, F32, F32])
    run_single(g, g.op("Where", cond, a, b))
    g2, (x, s, bias, mean, var) = graph_with([(2, 3, 2, 2), (3,), (3,), (3,), (3,)])
    inputs = ir.random_inputs(g2, seed=4)
    inputs["x4"] = np.abs(inputs["x4"]) + 0.5
    run_single(g2, g2.op("BatchNorm", x, s, bias, mean, var, epsilon=1e-3), inputs)


# -- injective -----------------------------------------------------------------------------------


def test_reshape_transpose_slice_expand_pad() -> None:
    g, (x,) = graph_with([(2, 3, 4)])
    run_single(g, g.op("Reshape", x, shape=(4, 6)))
    g, (x,) = graph_with([(2, 3, 4)])
    run_single(g, g.op("Transpose", x, perm=(2, 0, 1)))
    g, (x,) = graph_with([(5, 7)])
    run_single(g, g.op("Slice", x, starts=(1, -6), ends=(5, 7), steps=(2, 3)))
    g, (x,) = graph_with([(3, 1)])
    run_single(g, g.op("Expand", x, shape=(2, 3, 4)))
    g, (x,) = graph_with([(2, 3)])
    run_single(g, g.op("Pad", x, pads=(1, 0, 2, 3), value=1.5))


def test_concat() -> None:
    g, (a, b, c) = graph_with([(2, 1, 3), (2, 4, 3), (2, 2, 3)])
    out = g.add_node("Concat", [a, b, c], {"axis": 1}).output
    run_single(g, out)


@pytest.mark.parametrize("index_dtype", [DType.int64, DType.int32])
@pytest.mark.parametrize("axis", [0, 1])
def test_gather_with_runtime_indices(index_dtype: DType, axis: int) -> None:
    g, (data, idx) = graph_with([(4, 5), (2, 3)], [F32, index_dtype])
    inputs = {
        "x0": np.arange(20, dtype=np.float32).reshape(4, 5),
        "x1": np.array([[0, -1, 2], [3, -4, 1]], index_dtype.numpy),
    }
    run_single(g, g.op("Gather", data, idx, axis=axis), inputs)


# -- anchors --------------------------------------------------------------------------------


@settings(max_examples=60, deadline=None)
@given(
    st.integers(1, 2),
    st.integers(3, 7),
    st.integers(3, 7),
    st.sampled_from([(1, 2), (2, 3), (4, 4), (3, 1)]),
    st.integers(1, 3),
    st.integers(1, 3),
    st.integers(1, 2),
    st.integers(1, 2),
    st.tuples(st.integers(0, 1), st.integers(0, 1), st.integers(0, 1), st.integers(0, 1)),
    st.booleans(),
    st.booleans(),
    st.sampled_from(["NHWC", "NCHW"]),
)
def test_conv2d(
    n: int, h: int, w: int, channels: tuple[int, int], kh: int, kw: int, s: int, d: int,
    pads: tuple[int, int, int, int], depthwise: bool, bias: bool, layout: str,
) -> None:  # fmt: skip
    c, o = channels
    groups = 1
    if depthwise:
        o, groups = c, c
    if (h + pads[0] + pads[2]) < d * (kh - 1) + 1 or (w + pads[1] + pads[3]) < d * (kw - 1) + 1:
        return
    g = Graph("conv")
    shape = (n, h, w, c) if layout == "NHWC" else (n, c, h, w)
    x = g.add_input("x", TensorType(shape, F32))
    i = c // groups
    rng = np.random.default_rng(5)
    wshape = (kh, kw, i, o) if layout == "NHWC" else (o, i, kh, kw)
    weight = g.add_constant(rng.standard_normal(wshape).astype(np.float32), "w")
    ins = [g.op("Relu", x), weight]
    if bias:
        ins.append(g.add_constant(rng.standard_normal(o).astype(np.float32), "b"))
    y = g.add_node(
        "Conv2d", ins,
        {"strides": (s, 1), "pads": pads, "dilations": (d, 1), "groups": groups, "layout": layout},
    ).output  # fmt: skip
    g.set_outputs([g.op("Mul", g.op("Sigmoid", y), g.add_constant(np.float32(3.0), "k"))])
    (lowered,) = check_graph(g)
    assert lowered.epilogue_block is not None
    assert lowered.dims["G"] == groups
    assert lowered.dims["SH"] == s
    assert lowered.dims["PAD_BLOCK"] == 0
    if c > 1:
        assert lowered.pattern == ("depthwise_conv2d" if depthwise else "conv2d")
    # The same group with the padded input in a block of its own.
    (padded,) = check_graph(g, pad_block=True)
    roles = _roles(padded.func)
    has_pad = "pad" in roles.values()
    assert padded.dims["PAD_BLOCK"] == int(has_pad)
    if not has_pad:
        assert tir.structural_equal(padded.func, lowered.func)
        return
    assert any(p > 0 for p in pads)
    pad_name = next(name for name, role in roles.items() if role == "pad")
    assert padded.main_block is not None
    main_reads = tir.script(_block(padded.func, padded.main_block).body)
    assert "select" not in main_reads
    sch = Schedule(padded.func)
    sch.compute_inline(pad_name)
    assert tir.structural_equal(sch.func, lowered.func)


@pytest.mark.parametrize(
    ("a", "b"),
    [((5, 3), (3, 4)), ((2, 5, 3), (2, 3, 4)), ((2, 1, 5, 3), (3, 3, 4)), ((3, 5, 3), (3, 2))],
)
def test_matmul_with_batch_broadcast(a: tuple[int, ...], b: tuple[int, ...]) -> None:
    g, (x, y) = graph_with([a, b])
    out = g.op("MatMul", x, y)
    g.set_outputs([out])
    (lowered,) = check_graph(g)
    batch = math.prod(out.shape[:-2])
    assert lowered.dims == {"B": batch, "M": a[-2], "N": b[-1], "K": a[-1]}


@pytest.mark.parametrize("op", ["MaxPool2d", "AvgPool2d"])
@pytest.mark.parametrize("include_pad", [0, 1])
@pytest.mark.parametrize("layout", ["NHWC", "NCHW"])
def test_pool(op: str, include_pad: int, layout: str) -> None:
    shape = (1, 7, 6, 3) if layout == "NHWC" else (1, 3, 7, 6)
    g, (x,) = graph_with([shape])
    attrs: dict[str, object] = {"kernel": (3, 2), "strides": (2, 1), "pads": (1, 1, 1, 0),
                                "layout": layout}  # fmt: skip
    if op == "AvgPool2d":
        attrs["count_include_pad"] = include_pad
    elif include_pad:
        return
    out = g.add_node(op, [x], attrs).output  # type: ignore[arg-type]
    g.set_outputs([out])
    check_graph(g)


@pytest.mark.parametrize("op", ["ReduceSum", "ReduceMean", "ReduceMax"])
@pytest.mark.parametrize(("axes", "keepdims"), [((1,), 1), ((0, 2), 0), ((2,), 0), ((0, 1, 2), 1)])
def test_reductions(op: str, axes: tuple[int, ...], keepdims: int) -> None:
    g, (x,) = graph_with([(3, 4, 5)])
    out = g.op(op, x, axes=axes, keepdims=keepdims)
    g.set_outputs([out])
    (lowered,) = check_graph(g)
    assert lowered.dims["R"] == math.prod((3, 4, 5)[a] for a in axes)
    roles = _roles(lowered.func)
    assert roles[lowered.main_block] == "main"
    if op == "ReduceMean":
        assert lowered.epilogue_block is not None
        assert roles[lowered.epilogue_block] == "epilogue"
    else:
        assert lowered.epilogue_block is None


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_softmax(axis: int) -> None:
    g, (x,) = graph_with([(3, 4, 5)])
    out = g.op("Softmax", g.op("Mul", x, g.add_constant(np.float32(2.0), "scale")), axis=axis)
    g.set_outputs([out])
    (lowered,) = check_graph(g)
    roles = _roles(lowered.func)
    assert roles == {f"{lowered.main_block}_max": "max", f"{lowered.main_block}_sum": "sum",
                     lowered.main_block: "main"}  # fmt: skip


@pytest.mark.parametrize("axis", [1, 2])
def test_layernorm(axis: int) -> None:
    g, (x,) = graph_with([(2, 3, 4)])
    norm = (2, 3, 4)[axis:]
    rng = np.random.default_rng(6)
    scale = g.add_constant(rng.standard_normal(norm).astype(np.float32), "scale")
    bias = g.add_constant(rng.standard_normal(norm).astype(np.float32), "bias")
    out = g.op("LayerNorm", x, scale, bias, axis=axis, epsilon=1e-5)
    g.set_outputs([g.op("Gelu", out)])
    (lowered,) = check_graph(g)
    assert lowered.epilogue_block is not None
    assert set(_roles(lowered.func).values()) == {"mean", "var", "main", "epilogue"}


# -- groups ---------------------------------------------------------------------------------------


def _block(func: PrimFunc, name: str) -> Block:
    from tessera.tir.primitives.common import find_block, realize_of

    return realize_of(find_block(func, name)).block


def _roles(func: PrimFunc) -> dict[str, object]:
    from tessera.tir.primitives.common import block_paths, realize_of

    out: dict[str, object] = {}
    for p in block_paths(func):
        b = realize_of(p).block
        out[b.name] = dict(b.annotations).get(ROLE)
    return out


def test_small_network_with_fusion() -> None:
    g = Graph("net")
    rng = np.random.default_rng(7)
    x = g.add_input("image", TensorType((1, 8, 8, 3), F32))
    w1 = g.add_constant(rng.standard_normal((3, 3, 3, 4)).astype(np.float32), "w1")
    b1 = g.add_constant(rng.standard_normal(4).astype(np.float32), "b1")
    conv = g.op("Conv2d", x, w1, b1, pads=(1, 1, 1, 1), layout="NHWC")
    act = g.op("Relu", conv)
    dw = g.add_constant(rng.standard_normal((3, 3, 1, 4)).astype(np.float32), "dw")
    conv2 = g.op("Conv2d", act, dw, pads=(1, 1, 1, 1), strides=(2, 2), groups=4, layout="NHWC")
    res = g.op("Clip", conv2, min=0.0, max=6.0)
    pool = g.op("MaxPool2d", res, kernel=(2, 2), strides=(2, 2), layout="NHWC")
    flat = g.op("Reshape", pool, shape=(1, 16))
    fc = g.add_constant(rng.standard_normal((16, 5)).astype(np.float32), "fc")
    logits = g.op("MatMul", flat, fc)
    probs = g.op("Softmax", logits, axis=1)
    g.set_outputs([probs])
    lowered = check_graph(g)
    assert [lg.pattern for lg in lowered] == [
        "conv2d", "depthwise_conv2d", "pool", "matmul", "softmax",
    ]  # fmt: skip
    for lg in lowered:
        assert lg.main_block is not None
        assert _roles(lg.func)[lg.main_block] == "main"
    unfused = check_graph(g, fuse=False)
    assert len(unfused) > len(lowered)


def test_injective_group_has_no_main_block() -> None:
    g, (a, b) = graph_with([(3, 4), (4,)])
    out = g.op("Transpose", g.op("Mul", a, b), perm=(1, 0))
    g.set_outputs([out])
    (lowered,) = check_graph(g)
    assert (lowered.main_block, lowered.epilogue_block) == (None, None)
    assert lowered.dims == {"SIZE": 12}


def test_lowered_groups_can_be_scheduled() -> None:
    g = Graph("conv")
    rng = np.random.default_rng(8)
    x = g.add_input("x", TensorType((1, 6, 6, 4), F32))
    w = g.add_constant(rng.standard_normal((3, 3, 4, 8)).astype(np.float32), "w")
    y = g.op("Relu", g.op("Conv2d", x, w, pads=(1, 1, 1, 1), layout="NHWC"))
    g.set_outputs([y])
    values = values_of(g, ir.random_inputs(g, seed=2))
    from tessera import passes

    (group,) = passes.fuse(g)
    lowered = lower_group(group)
    s = Schedule(lowered.func)
    assert lowered.main_block is not None
    assert lowered.epilogue_block is not None
    _, oh, ow, o, kh, kw, ic = s.get_loops(lowered.main_block)
    oo, oi = s.split(o, [None, 4])
    s.reorder(oh, oo, ow, kh, kw, ic, oi)
    s.reverse_compute_at(lowered.epilogue_block, oo)
    s.decompose_reduction(lowered.main_block, ow)
    s.vectorize(oi)
    tir.verify(s.func)
    out = np.zeros((1, 6, 6, 8), np.float32)
    tir.interpret(s.func, [values["x"], values["w"], out], check_bounds=True)
    np.testing.assert_allclose(out, values[y.name], rtol=1e-4, atol=1e-4)


def test_buffer_names_stay_unique() -> None:
    g = Graph("names")
    x = g.add_input("sm_max", TensorType((2, 3), F32))
    out = g.op("Softmax", x, axis=1, name="sm")
    g.set_outputs([out])
    (lowered,) = check_graph(g)
    from tessera.tir.primitives.common import iter_paths
    from tessera.tir.stmt import Block

    names = [p.name for p in lowered.func.params]
    for path in iter_paths(lowered.func.body):
        if isinstance(path[-1], Block):
            names += [b.name for b in path[-1].alloc_buffers]
    assert len(names) == len(set(names))


def test_groups_of_an_optimized_model() -> None:
    from tessera import passes

    rng = np.random.default_rng(9)

    def const(shape: tuple[int, ...], name: str, low: float = 0.0) -> ir.Value:
        return g.add_constant((rng.random(shape) + low).astype(np.float32), name)

    g = Graph("model")
    x = g.add_input("x", TensorType((1, 3, 10, 10), F32))
    c = g.op("Conv2d", x, const((8, 3, 3, 3), "w", -0.5), pads=(1, 1, 1, 1), strides=(2, 2))
    stats = [const((8,), "s", 0.5), const((8,), "b"), const((8,), "m"), const((8,), "v", 0.5)]
    r = g.op("Relu", g.op("BatchNorm", c, *stats))
    c2 = g.op("Conv2d", r, const((8, 1, 3, 3), "dw", -0.5), pads=(1, 1, 1, 1), groups=8)
    p = g.op("MaxPool2d", g.op("Clip", c2, min=0.0, max=6.0), kernel=(2, 2), strides=(2, 2))
    flat = g.op("Reshape", g.op("ReduceMean", p, axes=(2, 3), keepdims=1), shape=(1, 8))
    logits = g.op("Add", g.op("MatMul", flat, const((8, 4), "fc", -0.5)), const((4,), "fb"))
    g.set_outputs([g.op("Softmax", logits, axis=1)])
    lowered = check_graph(passes.optimize(g))
    assert [lg.pattern for lg in lowered] == [
        "conv2d", "depthwise_conv2d", "pool", "reduce", "matmul", "softmax",
    ]  # fmt: skip


def _main_buffer_reads(lowered: LoweredGroup, block: str) -> list[tuple[str, ...]]:
    from tessera.tir.analysis import block_accesses
    from tessera.tir.primitives.common import find_block, realize_of

    b = realize_of(find_block(lowered.func, block)).block
    return [
        tuple(tir.script(i) for i in idx)
        for buf, accesses in block_accesses(b).loads.items()
        for idx in accesses
    ]


def test_batch_one_epilogue_reads_by_iteration_variable() -> None:
    from tessera import passes

    g = Graph("batch1")
    rng = np.random.default_rng(10)
    x = g.add_input("x", TensorType((1, 8, 8, 16), F32))
    w = g.add_constant(rng.standard_normal((1, 1, 16, 16)).astype(np.float32), "w")
    b = g.add_constant(rng.standard_normal(16).astype(np.float32), "b")
    conv = g.op("Conv2d", x, w, b, layout="NHWC")
    pool = g.op("MaxPool2d", x, kernel=(1, 1), layout="NHWC")
    out = g.op("Relu", g.op("Add", conv, pool))
    g.set_outputs([out])
    values = values_of(g, ir.random_inputs(g, seed=11))
    group = next(gr for gr in passes.fuse(g) if gr.pattern == "conv2d")
    lowered = check_group(group, values)
    assert lowered.epilogue_block is not None
    for idx in _main_buffer_reads(lowered, lowered.epilogue_block):
        assert idx[0] == "v_i0"
    s = Schedule(lowered.func)
    assert lowered.main_block is not None
    copy = s.cache_write(lowered.main_block, 0, "local")
    s.reverse_compute_inline(lowered.epilogue_block)
    assert s.blocks() == [lowered.main_block, copy]
    args = [values[v.name] for v in group.inputs]
    res = np.zeros(out.shape, np.float32)
    tir.interpret(s.func, [*args, res], check_bounds=True)
    np.testing.assert_allclose(res, values[out.name], rtol=1e-4, atol=1e-4)


def test_dense_layer_prologue_and_epilogue_read_by_iteration_variable() -> None:
    from tessera import passes

    g = Graph("dense")
    rng = np.random.default_rng(12)
    x = g.add_input("x", TensorType((1, 4, 8), F32))
    scale = g.add_constant(rng.standard_normal(8).astype(np.float32), "ln_scale")
    shift = g.add_constant(rng.standard_normal(8).astype(np.float32), "ln_bias")
    ln = g.op("LayerNorm", x, scale, shift, axis=2)
    w = g.add_constant(rng.standard_normal((8, 6)).astype(np.float32), "w")
    bias = g.add_constant(rng.standard_normal(6).astype(np.float32), "b")
    out = g.op("Gelu", g.op("Add", g.op("MatMul", ln, w), bias))
    g.set_outputs([out])
    values = values_of(g, ir.random_inputs(g, seed=13))
    group = next(gr for gr in passes.fuse(g) if gr.pattern == "matmul")
    lowered = check_group(group, values)
    assert lowered.main_block is not None
    assert lowered.epilogue_block is not None
    for block in (lowered.main_block, lowered.epilogue_block):
        for idx in _main_buffer_reads(lowered, block):
            assert "0" not in idx[:1]
    s = Schedule(lowered.func)
    s.cache_write(lowered.main_block, 0, "local")
    s.reverse_compute_inline(lowered.epilogue_block)
    res = np.zeros(out.shape, np.float32)
    tir.interpret(s.func, [*(values[v.name] for v in group.inputs), res], check_bounds=True)
    np.testing.assert_allclose(res, values[out.name], rtol=1e-4, atol=1e-4)


def test_padded_conv_staged_through_shared_memory() -> None:
    """The GPU pattern: the pad block computed per tile of output rows, copied into
    threadgroup memory cooperatively, read by threads that each own an output."""
    from tessera import passes

    g = Graph("conv")
    rng = np.random.default_rng(14)
    x = g.add_input("x", TensorType((1, 8, 7, 4), F32))
    w = g.add_constant(rng.standard_normal((3, 3, 4, 8)).astype(np.float32), "w")
    conv = g.op("Conv2d", x, w, pads=(1, 1, 1, 0), layout="NHWC")
    g.set_outputs([g.op("Relu", conv)])
    values = values_of(g, ir.random_inputs(g, seed=15))
    (group,) = passes.fuse(g)
    lowered = lower_group(group, pad_block=True)
    assert lowered.dims["PAD_BLOCK"] == 1
    main = lowered.main_block
    assert main is not None
    pad = next(name for name, role in _roles(lowered.func).items() if role == "pad")
    s = Schedule(lowered.func)
    _, oh, ow, _, _, _, _ = s.get_loops(main)
    oh_tile, oh_in = s.split(oh, [None, 3])  # 8 rows in tiles of 3: the last is partial
    reads = [r.buffer.name for r in _block(s.func, main).reads]
    shared = s.cache_read(main, reads.index(pad), "shared")
    s.compute_at(shared, oh_tile)
    s.compute_at(pad, oh_tile)
    s.bind(oh_tile, "threadgroup.x")
    s.bind(oh_in, "thread.y")
    s.bind(ow, "thread.x")
    copy_loops = s.get_loops(shared)
    s.bind(s.fuse(*copy_loops[copy_loops.index(oh_tile) + 1 :]), "thread.x")
    tir.verify(s.func)
    expected = values[group.output.name]
    for func in (s.func, tir.compact_buffers(s.func)):
        out = np.full(group.output.shape, np.nan, np.float32)
        args = [np.array(values[v.name], order="C") for v in group.inputs]
        tir.interpret(func, [*args, out], check_bounds=True)
        np.testing.assert_allclose(out, expected, rtol=1e-4, atol=1e-4)
    compact = tir.compact_buffers(s.func)
    text = tir.script(compact)
    assert "alloc_buffer(float32[1, 5, 8, 4], 'shared')" in text
