"""Hypothesis strategies for random graphs, and helpers that run fused graphs.

``graphs()`` draws small dataflow graphs from the operator set: elementwise
chains, broadcasting with constants, matrix products, convolutions, pooling,
reductions, layout changes and (optionally) the ONNX-only operators. Shapes
stay small so the interpreter runs each example in milliseconds.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import prod
from typing import Any

import numpy as np
from hypothesis import strategies as st

from tessera.dtype import DType
from tessera.ir import Graph, TensorType, Value, interpret
from tessera.plan import FusedGroup, MemoryPlan, StorageKind

type Array = np.ndarray[Any, Any]
type Draw = Callable[[st.SearchStrategy[Any]], Any]

F32 = DType.float32


def _const(
    draw: Draw, g: Graph, shape: tuple[int, ...], name: str, positive: bool = False
) -> Value:
    seed = draw(st.integers(0, 2**16))
    data = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    if positive:
        data = np.abs(data) + 0.5
    return g.add_constant(data, name)


def _suffix_shape(draw: Draw, shape: tuple[int, ...]) -> tuple[int, ...]:
    """A shape that broadcasts against ``shape`` (a suffix, some dims set to 1)."""
    if not shape:
        return ()
    k = draw(st.integers(0, len(shape)))
    tail = shape[len(shape) - k :]
    return tuple(1 if draw(st.booleans()) else d for d in tail)


def _step(draw: Draw, g: Graph, pool: list[Value], import_ops: bool) -> Value | None:
    x = draw(st.sampled_from(pool[-4:]))
    s = x.shape
    choice = draw(
        st.sampled_from(
            [
                "unary",
                "unary",
                "binary_const",
                "binary_pool",
                "matmul",
                "transpose",
                "reshape",
                "reduce",
                "softmax",
                "layernorm",
                "conv",
                "pool",
                "slice",
                "concat",
                "pad",
                "decomposed",
                "decomposed",
                "rows_gemm",
            ]
            + (
                ["gemm", "flatten", "squeeze", "identity", "batchnorm", "conv_bn", "rows_gemm"]
                if import_ops
                else []
            )
        )
    )
    match choice:
        case "unary":
            op = draw(st.sampled_from(["Relu", "Tanh", "Sigmoid", "Neg", "Abs", "Silu", "Gelu"]))
            if op == "Gelu":
                return g.op(op, x, approximate=draw(st.sampled_from(["none", "tanh"])))
            return g.op(op, x)
        case "binary_const":
            op = draw(st.sampled_from(["Add", "Sub", "Mul", "Maximum", "Minimum"]))
            c = _const(draw, g, _suffix_shape(draw, s), "c")
            return g.op(op, x, c) if draw(st.booleans()) else g.op(op, c, x)
        case "binary_pool":
            same = [v for v in pool if v.shape == s and v is not x]
            if not same:
                return None
            op = draw(st.sampled_from(["Add", "Mul", "Sub"]))
            return g.op(op, x, draw(st.sampled_from(same)))
        case "matmul":
            if len(s) < 2:
                return None
            w = _const(draw, g, (s[-1], draw(st.integers(1, 6))), "w")
            return g.op("MatMul", x, w)
        case "transpose":
            perm = draw(st.permutations(range(len(s))))
            return g.op("Transpose", x, perm=tuple(perm))
        case "reshape":
            if len(s) >= 2 and draw(st.booleans()):
                return g.op("Reshape", x, shape=(*s[:-2], s[-2] * s[-1]))
            return g.op("Reshape", x, shape=(1, *s))
        case "reduce":
            if not s:
                return None
            op = draw(st.sampled_from(["ReduceSum", "ReduceMean", "ReduceMax"]))
            axes = draw(st.sets(st.integers(0, len(s) - 1), min_size=1))
            return g.op(op, x, axes=tuple(sorted(axes)), keepdims=draw(st.sampled_from([0, 1])))
        case "softmax":
            if not s:
                return None
            return g.op("Softmax", x, axis=draw(st.integers(-len(s), len(s) - 1)))
        case "layernorm":
            if not s:
                return None
            return g.op(
                "LayerNorm", x, _const(draw, g, s[-1:], "gamma"), _const(draw, g, s[-1:], "beta")
            )
        case "conv":
            if len(s) != 4 or min(s[2:]) < 2:
                return None
            k = draw(st.integers(1, 3))
            pad = draw(st.integers(0, k - 1))
            if min(s[2:]) + 2 * pad < k:
                return None
            o = draw(st.integers(1, 5))
            depthwise = draw(st.booleans())
            groups = s[1] if depthwise else 1
            o = s[1] if depthwise else o
            w = _const(draw, g, (o, s[1] // groups, k, k), "w")
            args = [x, w] + ([_const(draw, g, (o,), "b")] if draw(st.booleans()) else [])
            return g.op(
                "Conv2d", *args, pads=(pad, pad), strides=draw(st.integers(1, 2)), groups=groups
            )
        case "pool":
            if len(s) != 4 or min(s[2:]) < 2:
                return None
            op = draw(st.sampled_from(["MaxPool2d", "AvgPool2d"]))
            return g.op(op, x, kernel=2, strides=draw(st.integers(1, 2)))
        case "slice":
            if not s or s[0] < 2:
                return None
            return g.op("Slice", x, starts=(1,), ends=(s[0],), axes=(0,))
        case "concat":
            if not s:
                return None
            return g.op("Concat", x, x, axis=draw(st.integers(0, len(s) - 1)))
        case "pad":
            return g.op("Pad", x, pads=tuple(draw(st.integers(0, 1)) for _ in range(2 * len(s))))
        case "decomposed":
            if not s:
                return None
            return _decomposed(draw, g, x)
        case "rows_gemm":
            # An exporter's Linear on a 3-D input: merge rows, Gemm/MatMul, split rows again.
            if len(s) < 3:
                return None
            rows = g.op("Reshape", x, shape=(prod(s[:-1]), s[-1]))
            n = draw(st.integers(1, 5))
            y = g.op("MatMul", rows, _const(draw, g, (s[-1], n), "w"))
            y = g.op("Add", y, _const(draw, g, (n,), "b"))
            return g.op("Reshape", y, shape=(*s[:-1], n))
        case "conv_bn":
            if len(s) != 4 or min(s[2:]) < 3:
                return None
            groups = 2 if s[1] % 2 == 0 and draw(st.booleans()) else 1
            o = 2 * draw(st.integers(1, 3))
            y = g.op("Conv2d", x, _const(draw, g, (o, s[1] // groups, 3, 3), "w"), groups=groups)
            params = [_const(draw, g, (o,), n) for n in ("scale", "shift", "mean")]
            return g.op("BatchNorm", y, *params, _const(draw, g, (o,), "var", positive=True))
        case "gemm":
            if len(s) != 2:
                return None
            trans_b = draw(st.sampled_from([0, 1]))
            n = draw(st.integers(1, 5))
            w = _const(draw, g, (n, s[1]) if trans_b else (s[1], n), "w")
            bias = _const(draw, g, (n,), "bias")
            return g.op("Gemm", x, w, bias, transB=trans_b, alpha=draw(st.sampled_from([1.0, 0.5])))
        case "flatten":
            return g.op("Flatten", x, axis=draw(st.integers(0, len(s))))
        case "squeeze":
            return g.op("Squeeze", g.op("Unsqueeze", x, axes=(0,)), axes=(0,))
        case "identity":
            return g.op(draw(st.sampled_from(["Identity", "Dropout"])), x)
        case "batchnorm":
            if len(s) < 2:
                return None
            c = s[1]
            params = [_const(draw, g, (c,), n) for n in ("scale", "shift", "mean")]
            var = _const(draw, g, (c,), "var", positive=True)
            return g.op("BatchNorm", x, *params, var)
    return None


def _decomposed(draw: Draw, g: Graph, x: Value) -> Value:
    """A LayerNorm, GELU or SiLU spelled out with primitive operators, as exporters emit them."""
    kind = draw(st.sampled_from(["layernorm", "gelu", "gelu_tanh", "silu"]))
    if kind == "silu":
        return g.op("Mul", x, g.op("Sigmoid", x))
    if kind == "gelu_tanh":
        cube = g.op("Pow", x, g.add_constant(np.float32(3.0), "three"))
        inner = g.op("Add", x, g.op("Mul", cube, g.add_constant(np.float32(0.044715), "k")))
        scale = g.add_constant(np.float32(np.sqrt(2 / np.pi)), "c")
        t = g.op(
            "Add", g.op("Tanh", g.op("Mul", scale, inner)), g.add_constant(np.float32(1), "one")
        )
        return g.op("Mul", g.op("Mul", g.add_constant(np.float32(0.5), "half"), x), t)
    if kind == "layernorm":
        axis = len(x.shape) - 1
        mean = g.op("ReduceMean", x, axes=(axis,))
        centered = g.op("Sub", x, mean)
        two = g.add_constant(np.float32(2.0), "two")
        var = g.op("ReduceMean", g.op("Pow", centered, two), axes=(axis,))
        eps = g.add_constant(np.float32(1e-5), "eps")
        normed = g.op("Div", centered, g.op("Sqrt", g.op("Add", var, eps)))
        gamma = _const(draw, g, x.shape[-1:], "gamma")
        beta = _const(draw, g, x.shape[-1:], "beta")
        return g.op("Add", g.op("Mul", normed, gamma), beta)
    sqrt2 = g.add_constant(np.float32(np.sqrt(2.0)), "sqrt2")
    one = g.add_constant(np.float32(1.0), "one")
    half = g.add_constant(np.float32(0.5), "half")
    erf = g.op("Erf", g.op("Div", x, sqrt2))
    return g.op("Mul", g.op("Mul", x, g.op("Add", erf, one)), half)


@st.composite
def graphs(draw: Draw, *, import_ops: bool = False, max_nodes: int = 12) -> Graph:
    """A random graph with one or two inputs and one to three outputs."""
    g = Graph("random")
    rank = draw(st.sampled_from([2, 3, 4, 4]))
    dims = [draw(st.integers(1, 3))] + [draw(st.integers(2, 5)) for _ in range(rank - 1)]
    x = g.add_input("x", TensorType(tuple(dims), F32))
    pool = [x]
    if draw(st.booleans()):
        pool.append(g.add_input("y", TensorType(tuple(dims), F32)))
    for _ in range(draw(st.integers(2, max_nodes))):
        if len(g.nodes) >= max_nodes * 3:
            break
        out = _step(draw, g, pool, import_ops)
        if out is not None and out.type.numel <= 4096 and prod(out.shape) > 0:
            pool.append(out)
    produced = [v for v in pool if v.producer is not None]
    if not produced:
        produced = [g.op("Relu", x)]
    outputs = [produced[-1]]
    for v in draw(st.lists(st.sampled_from(produced), max_size=2)):
        if v not in outputs:
            outputs.append(v)
    g.set_outputs(outputs)
    g.verify()
    return g


def run_groups(
    graph: Graph, groups: Sequence[FusedGroup], inputs: Mapping[str, Array]
) -> dict[str, Array]:
    """Run each group as its own graph, in order, passing values between them by name."""
    env: dict[int, Array] = {id(v): np.asarray(inputs[v.name]) for v in graph.inputs}
    for group in groups:
        sub = Graph(group.name)
        feeds: dict[str, Array] = {}
        mapping: dict[int, Value] = {}
        for v in group.inputs:
            if v.is_constant:
                mapping[id(v)] = v
            else:
                mapping[id(v)] = sub.add_input(v.name, v.type)
                feeds[sub.inputs[-1].name] = env[id(v)]
        for node in group.nodes:
            args = [mapping[id(v)] for v in node.inputs]
            mapping[id(node.output)] = sub.op(node.op, *args, **node.attrs)
        sub.set_outputs([mapping[id(group.output)]])
        env[id(group.output)] = interpret(sub, feeds)[sub.outputs[0].name]
    return {v.name: env[id(v)] if not v.is_constant else np.asarray(v.data) for v in graph.outputs}


def run_plan(
    graph: Graph, groups: Sequence[FusedGroup], plan: MemoryPlan, inputs: Mapping[str, Array]
) -> dict[str, Array]:
    """Run the groups against real byte buffers placed as ``plan`` says.

    Every kernel reads its inputs from and writes its output to the bytes of
    its ``BufferRef``, so overlapping live values would corrupt the result.
    """
    arena = np.full(plan.arena_bytes, 0xAB, np.uint8)
    constants = plan.constants.copy()
    outs = [np.zeros(v.type.nbytes, np.uint8) for v in graph.outputs]
    ins = [np.ascontiguousarray(inputs[v.name]).reshape(-1).view(np.uint8) for v in graph.inputs]

    def buffer(name: str) -> Array:
        ref = plan.refs[name]
        match ref.kind:
            case StorageKind.INPUT:
                return ins[ref.index]
            case StorageKind.OUTPUT:
                return outs[ref.index]
            case StorageKind.CONSTANT:
                return constants[ref.offset : ref.offset + ref.nbytes]
            case StorageKind.ARENA:
                return arena[ref.offset : ref.offset + ref.nbytes]

    def load(v: Value) -> Array:
        return buffer(v.name).view(v.dtype.numpy).reshape(v.shape).copy()

    for group in groups:
        if group.is_view:
            # A view shares its input's bytes; when the plan could not alias them
            # (a view of an input returned as an output), the caller copies.
            if plan.refs[group.output.name] != plan.refs[group.inputs[0].name]:
                buffer(group.output.name)[:] = buffer(group.inputs[0].name)
            continue
        result = run_groups(_group_graph(group), [group], {v.name: load(v) for v in group.inputs})
        data = np.ascontiguousarray(next(iter(result.values()))).reshape(-1).view(np.uint8)
        buffer(group.output.name)[:] = data
    return {v.name: load(v) for v in graph.outputs}


def _group_graph(group: FusedGroup) -> Graph:
    """A graph whose inputs are the group's non-constant inputs and output is its output."""
    g = Graph(group.name)
    g.inputs = [v for v in group.inputs if not v.is_constant]
    g.nodes = list(group.nodes)
    g.outputs = [group.output]
    return g


def assert_same_outputs(before: Graph, after: Graph, seed: int = 2) -> None:
    """``after`` computes ``before``'s outputs about as accurately as ``before`` itself.

    Both run in float32 against a float64 run of ``before``; the error of
    ``after`` may exceed the error of ``before`` by a few times (rewrites
    reorder rounding) or by 2e-5 of the output's magnitude.
    """
    from tessera.ir import random_inputs

    feeds = random_inputs(before, seed=seed)
    exact = interpret(before, feeds, float64=True)
    plain = interpret(before, feeds)
    got = interpret(after, feeds)
    assert len(got) == len(exact)
    for out, (name, ref) in zip(after.outputs, exact.items(), strict=True):
        value = got[out.name]
        assert value.shape == ref.shape
        if ref.dtype.kind in "biu":
            np.testing.assert_array_equal(value, ref)
            continue
        scale = max(1.0, float(np.abs(ref).max()))
        error = float(np.abs(value - ref).max())
        baseline = float(np.abs(plain[name] - ref).max())
        assert error <= max(4 * baseline, 2e-5 * scale), (name, error, baseline)
