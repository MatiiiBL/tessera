"""Lowered groups for the template tests, shaped like ``compute.lower_group``'s
output: prologue operators inlined into the loads of the main block, the
anchor or reduction as the main block, and at most one epilogue block. Each
comes with float64 NumPy references for its inputs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from tessera import te
from tessera.dtype import DType
from tessera.ir.graph import Graph, TensorType
from tessera.plan import FusedGroup, LoweredGroup
from tessera.tir.expr import Expr, call, ge, logical_and, lt, maximum, minimum, select
from tessera.tir.stmt import PrimFunc

type NDArray = npt.NDArray[Any]
F32 = DType.float32


@dataclass
class Workload:
    lowered: LoweredGroup
    inputs: list[NDArray]
    reference: Callable[..., NDArray]
    tolerance: float = 1e-4
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def func(self) -> PrimFunc:
        return self.lowered.func

    def expected(self) -> NDArray:
        return self.reference(*(x.astype(np.float64) for x in self.inputs))

    def output(self) -> NDArray:
        out = self.func.params[-1]
        return np.full(out.shape, np.nan, out.dtype.numpy)


def _rand(rng: np.random.Generator, *shape: int) -> NDArray:
    return rng.standard_normal(shape).astype(np.float32)


def _group(name: str, pattern: str, build: Callable[[Graph], None]) -> FusedGroup:
    """A FusedGroup over the nodes ``build`` adds to a fresh graph (the anchor is
    the first node of kind REDUCTION or ANCHOR)."""
    from tessera.ir.ops import OpKind

    g = Graph(name)
    build(g)
    nodes = list(g.nodes)
    anchor = next((n for n in nodes if n.opdef.kind in (OpKind.REDUCTION, OpKind.ANCHOR)), None)
    produced = {n.output for n in nodes}
    inputs: list[Any] = []
    for n in nodes:
        for v in n.inputs:
            if v not in produced and v not in inputs:
                inputs.append(v)
    return FusedGroup(name, nodes, inputs, nodes[-1].output, anchor, pattern)


def _clamp(e: Expr, n: int) -> Expr:
    return minimum(maximum(e, 0), n - 1)


def matmul(
    m: int, n: int, k: int, *, batch: int = 0, epilogue: bool = True, seed: int = 0
) -> Workload:
    """relu(A @ B + bias) (or A @ B), A of shape [batch, m, k] when batch > 0."""
    lead = (batch,) if batch else ()
    a = te.placeholder((*lead, m, k), F32, "A")
    b = te.placeholder((k, n), F32, "B")
    bias = te.placeholder((n,), F32, "bias")
    r = te.reduce_axis(k, "k")
    names = [*(["b"] if batch else []), "i", "j"]

    def body(*ix: Expr) -> Any:
        *bs, i, j = ix
        return te.sum(a[(*bs, i, r)] * b[r, j], axis=r)

    mm = te.compute((*lead, m, n), body, "matmul", axis_names=names)
    params = [a, b]
    out = mm
    if epilogue:
        out = te.compute(
            (*lead, m, n),
            lambda *ix: maximum(mm[ix] + bias[ix[-1]], 0.0),
            "relu",
            axis_names=names,
        )
        params.append(bias)

    def build(g: Graph) -> None:
        x = g.add_input("A", TensorType((*lead, m, k), F32))
        w = g.add_constant(np.zeros((k, n), np.float32), "B")
        y = g.op("MatMul", x, w)
        if epilogue:
            y = g.op("Relu", g.op("Add", y, g.add_constant(np.zeros(n, np.float32), "bias")))

    rng = np.random.default_rng(seed)
    inputs = [_rand(rng, *lead, m, k), _rand(rng, k, n)]
    if epilogue:
        inputs.append(_rand(rng, n))

    def reference(x: NDArray, w: NDArray, *rest: NDArray) -> NDArray:
        y = x @ w
        return np.maximum(y + rest[0], 0) if rest else y

    func = te.create_prim_func([*params, out], "matmul_group")
    dims = {"B": max(batch, 1), "M": m, "N": n, "K": k}
    lowered = LoweredGroup(
        _group("matmul_group", "matmul", build),
        func,
        "matmul",
        "relu" if epilogue else None,
        dims,
    )
    return Workload(lowered, inputs, reference)


def _conv_out(size: int, k: int, s: int, d: int, p0: int, p1: int) -> int:
    return (size + p0 + p1 - d * (k - 1) - 1) // s + 1


def conv2d(
    n: int,
    h: int,
    w: int,
    c: int,
    o: int,
    kernel: tuple[int, int] = (3, 3),
    strides: tuple[int, int] = (1, 1),
    pads: tuple[int, int, int, int] = (1, 1, 1, 1),
    dilations: tuple[int, int] = (1, 1),
    *,
    depthwise: bool = False,
    epilogue: bool = True,
    seed: int = 0,
) -> Workload:
    """NHWC convolution with HWIO weights, padding inlined into the loads as a
    Select, then relu(y + bias)."""
    (kh, kw), (sh, sw), (dh, dw) = kernel, strides, dilations
    pt, pl, pb, pr = pads
    oh, ow = _conv_out(h, kh, sh, dh, pt, pb), _conv_out(w, kw, sw, dw, pl, pr)
    if depthwise:
        o = c
    x = te.placeholder((n, h, w, c), F32, "X")
    wt = te.placeholder((kh, kw, 1 if depthwise else c, o), F32, "W")
    bias = te.placeholder((o,), F32, "bias")
    rh, rw = te.reduce_axis(kh, "rh"), te.reduce_axis(kw, "rw")
    rc = te.reduce_axis(c, "rc")

    def load(b: Expr, y: Expr, xx: Expr, ch: Expr) -> Expr:
        ih, iw = y * sh + rh * dh - pt, xx * sw + rw * dw - pl
        inside = logical_and(logical_and(ge(ih, 0), lt(ih, h)), logical_and(ge(iw, 0), lt(iw, w)))
        return select(inside, x[b, _clamp(ih, h), _clamp(iw, w), ch], 0.0)

    if depthwise:
        conv = te.compute(
            (n, oh, ow, o),
            lambda b, y, xx, ch: te.sum(load(b, y, xx, ch) * wt[rh, rw, 0, ch], axis=[rh, rw]),
            "conv",
            axis_names=["n", "oh", "ow", "o"],
        )
    else:
        conv = te.compute(
            (n, oh, ow, o),
            lambda b, y, xx, ch: te.sum(load(b, y, xx, rc) * wt[rh, rw, rc, ch], axis=[rh, rw, rc]),
            "conv",
            axis_names=["n", "oh", "ow", "o"],
        )
    params = [x, wt]
    out = conv
    if epilogue:
        out = te.compute(
            (n, oh, ow, o),
            lambda *ix: maximum(conv[ix] + bias[ix[-1]], 0.0),
            "relu",
            axis_names=["n", "oh", "ow", "o"],
        )
        params.append(bias)
    attrs = {
        "strides": strides,
        "pads": pads,
        "dilations": dilations,
        "groups": c if depthwise else 1,
        "layout": "NHWC",
    }

    def build(g: Graph) -> None:
        xv = g.add_input("X", TensorType((n, h, w, c), F32))
        wv = g.add_constant(np.zeros((kh, kw, 1 if depthwise else c, o), np.float32), "W")
        y = g.op("Conv2d", xv, wv, **attrs)  # type: ignore[arg-type]
        if epilogue:
            y = g.op("Relu", g.op("Add", y, g.add_constant(np.zeros(o, np.float32), "bias")))

    def reference(xs: NDArray, ws: NDArray, *rest: NDArray) -> NDArray:
        padded = np.pad(xs, ((0, 0), (pt, pb), (pl, pr), (0, 0)))
        y = np.zeros((n, oh, ow, o))
        for i in range(kh):
            for j in range(kw):
                rows = slice(i * dh, i * dh + sh * (oh - 1) + 1, sh)
                cols = slice(j * dw, j * dw + sw * (ow - 1) + 1, sw)
                patch = padded[:, rows, cols, :]
                y += patch * ws[i, j, 0] if depthwise else patch @ ws[i, j]
        return np.maximum(y + rest[0], 0) if rest else y

    rng = np.random.default_rng(seed)
    inputs = [_rand(rng, n, h, w, c), _rand(rng, kh, kw, 1 if depthwise else c, o)]
    if epilogue:
        inputs.append(_rand(rng, o))
    pattern = "depthwise_conv2d" if depthwise else "conv2d"
    dims = {
        "N": n, "H": h, "W": w, "C": c, "KH": kh, "KW": kw, "O": o,
        "OH": oh, "OW": ow, "SH": sh, "SW": sw,
    }  # fmt: skip
    lowered = LoweredGroup(
        _group("conv_group", pattern, build),
        te.create_prim_func([*params, out], "conv_group"),
        "conv",
        "relu" if epilogue else None,
        dims,
    )
    return Workload(lowered, inputs, reference)


def pool(
    kind: str,
    n: int,
    h: int,
    w: int,
    c: int,
    kernel: tuple[int, int] = (3, 3),
    strides: tuple[int, int] = (2, 2),
    pads: tuple[int, int, int, int] = (1, 1, 1, 1),
    seed: int = 0,
) -> Workload:
    """NHWC MaxPool2d or AvgPool2d (padding counted), as a reduction over the window."""
    (kh, kw), (sh, sw) = kernel, strides
    pt, pl, pb, pr = pads
    oh, ow = _conv_out(h, kh, sh, 1, pt, pb), _conv_out(w, kw, sw, 1, pl, pr)
    x = te.placeholder((n, h, w, c), F32, "X")
    rh, rw = te.reduce_axis(kh, "rh"), te.reduce_axis(kw, "rw")
    pad_value = -np.inf if kind == "max" else 0.0

    def load(b: Expr, y: Expr, xx: Expr, ch: Expr) -> Expr:
        ih, iw = y * sh + rh - pt, xx * sw + rw - pl
        inside = logical_and(logical_and(ge(ih, 0), lt(ih, h)), logical_and(ge(iw, 0), lt(iw, w)))
        return select(inside, x[b, _clamp(ih, h), _clamp(iw, w), ch], float(pad_value))

    reducer = te.max if kind == "max" else te.sum
    main = te.compute(
        (n, oh, ow, c),
        lambda b, y, xx, ch: reducer(load(b, y, xx, ch), axis=[rh, rw]),
        "pool",
        axis_names=["n", "oh", "ow", "c"],
    )
    out = main
    if kind == "avg":
        out = te.compute(
            (n, oh, ow, c),
            lambda *ix: main[ix] * (1.0 / (kh * kw)),
            "average",
            axis_names=list("nhwc"),
        )
    op = "MaxPool2d" if kind == "max" else "AvgPool2d"
    attrs: dict[str, Any] = {"kernel": kernel, "strides": strides, "pads": pads, "layout": "NHWC"}
    if kind == "avg":
        attrs["count_include_pad"] = 1

    def build(g: Graph) -> None:
        g.op(op, g.add_input("X", TensorType((n, h, w, c), F32)), **attrs)

    def reference(xs: NDArray) -> NDArray:
        padded = np.pad(xs, ((0, 0), (pt, pb), (pl, pr), (0, 0)), constant_values=pad_value)
        windows = [
            padded[:, i : i + sh * (oh - 1) + 1 : sh, j : j + sw * (ow - 1) + 1 : sw]
            for i in range(kh)
            for j in range(kw)
        ]
        stacked = np.stack(windows)
        return stacked.max(axis=0) if kind == "max" else stacked.mean(axis=0)

    rng = np.random.default_rng(seed)
    lowered = LoweredGroup(
        _group("pool_group", "pool", build),
        te.create_prim_func([x, out], "pool_group"),
        "pool",
        "average" if kind == "avg" else None,
        {"N": n, "H": h, "W": w, "C": c, "KH": kh, "KW": kw, "OH": oh, "OW": ow},
    )
    return Workload(lowered, [_rand(rng, n, h, w, c)], reference)


def reduce(kind: str, shape: tuple[int, ...], axes: tuple[int, ...], seed: int = 0) -> Workload:
    """ReduceSum / ReduceMean / ReduceMax over ``axes`` (contiguous), keepdims=0."""
    assert list(axes) == list(range(axes[0], axes[-1] + 1))
    x = te.placeholder(shape, F32, "X")
    outer, inner = shape[: axes[0]], shape[axes[-1] + 1 :]
    raxes = [te.reduce_axis(shape[a], f"r{a}") for a in axes]
    reducer = te.max if kind == "max" else te.sum

    def body(*ix: Expr) -> Any:
        o, i = ix[: len(outer)], ix[len(outer) :]
        return reducer(x[(*o, *raxes, *i)], axis=raxes)

    out_shape = (*outer, *inner)
    names = [f"d{i}" for i in range(len(out_shape))]
    main = te.compute(out_shape, body, "reduce", axis_names=names)
    out = main
    count = int(np.prod([shape[a] for a in axes]))
    if kind == "mean":
        out = te.compute(out_shape, lambda *ix: main[ix] * (1.0 / count), "mean", axis_names=names)
    op = {"sum": "ReduceSum", "mean": "ReduceMean", "max": "ReduceMax"}[kind]

    def build(g: Graph) -> None:
        g.op(op, g.add_input("X", TensorType(shape, F32)), axes=axes, keepdims=0)

    def reference(xs: NDArray) -> NDArray:
        fn = {"sum": np.sum, "mean": np.mean, "max": np.max}[kind]
        return fn(xs, axis=axes)

    rng = np.random.default_rng(seed)
    dims = {
        "OUTER": int(np.prod(outer)),
        "R": count,
        "INNER": int(np.prod(inner)),
    }
    lowered = LoweredGroup(
        _group("reduce_group", "reduce", build),
        te.create_prim_func([x, out], "reduce_group"),
        "reduce",
        "mean" if kind == "mean" else None,
        dims,
    )
    return Workload(lowered, [_rand(rng, *shape)], reference)


def softmax(m: int, n: int, *, scale: bool = False, seed: int = 0) -> Workload:
    """Softmax over the last axis: row max, sum of exponentials, normalization
    (optionally of x * 0.125, a prologue inlined into every load)."""
    x = te.placeholder((m, n), F32, "X")

    def value(i: Expr, j: Expr) -> Expr:
        return x[i, j] * 0.125 if scale else x[i, j]

    k1, k2 = te.reduce_axis(n, "k"), te.reduce_axis(n, "k")
    mx = te.compute((m,), lambda i: te.max(value(i, k1), axis=k1), "max", axis_names=["i"])
    sm = te.compute(
        (m,), lambda i: te.sum(call("exp", value(i, k2) - mx[i]), axis=k2), "sum", axis_names=["i"]
    )
    out = te.compute(
        (m, n),
        lambda i, j: call("exp", value(i, j) - mx[i]) / sm[i],
        "softmax",
        axis_names=["i", "j"],
    )

    def build(g: Graph) -> None:
        g.op("Softmax", g.add_input("X", TensorType((m, n), F32)), axis=1)

    def reference(xs: NDArray) -> NDArray:
        v = xs * 0.125 if scale else xs
        e = np.exp(v - v.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    rng = np.random.default_rng(seed)
    lowered = LoweredGroup(
        _group("softmax_group", "softmax", build),
        te.create_prim_func([x, out], "softmax_group"),
        "softmax",
        None,
        {"OUTER": m, "R": n, "INNER": 1},
    )
    return Workload(lowered, [_rand(rng, m, n) * 3], reference)


def layernorm(m: int, n: int, *, eps: float = 1e-5, seed: int = 0) -> Workload:
    """LayerNorm over the last axis with scale and bias: mean, variance, normalization."""
    x = te.placeholder((m, n), F32, "X")
    gamma = te.placeholder((n,), F32, "scale")
    beta = te.placeholder((n,), F32, "bias")
    k1, k2 = te.reduce_axis(n, "k"), te.reduce_axis(n, "k")
    mean = te.compute(
        (m,), lambda i: te.sum(x[i, k1] * (1.0 / n), axis=k1), "mean", axis_names=["i"]
    )
    var = te.compute(
        (m,),
        lambda i: te.sum((x[i, k2] - mean[i]) * (x[i, k2] - mean[i]) * (1.0 / n), axis=k2),
        "var",
        axis_names=["i"],
    )
    out = te.compute(
        (m, n),
        lambda i, j: (x[i, j] - mean[i]) * call("rsqrt", var[i] + eps) * gamma[j] + beta[j],
        "layernorm",
        axis_names=["i", "j"],
    )

    def build(g: Graph) -> None:
        xv = g.add_input("X", TensorType((m, n), F32))
        s = g.add_constant(np.ones(n, np.float32), "scale")
        b = g.add_constant(np.zeros(n, np.float32), "bias")
        g.op("LayerNorm", xv, s, b, axis=1, epsilon=eps)

    def reference(xs: NDArray, gs: NDArray, bs: NDArray) -> NDArray:
        mu = xs.mean(axis=1, keepdims=True)
        sd = np.sqrt(((xs - mu) ** 2).mean(axis=1, keepdims=True) + eps)
        return (xs - mu) / sd * gs + bs

    rng = np.random.default_rng(seed)
    lowered = LoweredGroup(
        _group("layernorm_group", "layernorm", build),
        te.create_prim_func([x, gamma, beta, out], "layernorm_group"),
        "layernorm",
        None,
        {"OUTER": m, "R": n, "INNER": 1},
    )
    return Workload(lowered, [_rand(rng, m, n), _rand(rng, n), _rand(rng, n)], reference)


def injective(n: int, h: int, w: int, c: int, seed: int = 0) -> Workload:
    """A transpose of NHWC to NCHW of sigmoid(x) * y + 1, broadcast over channels:
    one block, everything inlined."""
    x = te.placeholder((n, h, w, c), F32, "X")
    y = te.placeholder((c,), F32, "Y")
    out = te.compute(
        (n, c, h, w),
        lambda b, ch, i, j: call("sigmoid", x[b, i, j, ch]) * y[ch] + 1.0,
        "out",
        axis_names=["n", "c", "h", "w"],
    )

    def build(g: Graph) -> None:
        xv = g.add_input("X", TensorType((n, h, w, c), F32))
        yv = g.add_constant(np.ones(c, np.float32), "Y")
        g.op(
            "Transpose",
            g.op(
                "Add",
                g.op("Mul", g.op("Sigmoid", xv), yv),
                g.add_constant(np.ones(1, np.float32), "one"),
            ),
            perm=(0, 3, 1, 2),
        )

    def reference(xs: NDArray, ys: NDArray) -> NDArray:
        return (1 / (1 + np.exp(-xs)) * ys + 1).transpose(0, 3, 1, 2)

    rng = np.random.default_rng(seed)
    lowered = LoweredGroup(
        _group("injective_group", "injective", build),
        te.create_prim_func([x, y, out], "injective_group"),
        None,
        None,
        {},
    )
    return Workload(lowered, [_rand(rng, n, h, w, c), _rand(rng, c)], reference)


def elementwise(shape: tuple[int, ...], seed: int = 0) -> Workload:
    """gelu(x) * 2 + b, broadcast over the last axis (one block)."""
    x = te.placeholder(shape, F32, "X")
    b = te.placeholder((shape[-1],), F32, "b")

    def gelu(v: Expr) -> Expr:
        return v * 0.5 * (call("erf", v * 0.7071067811865476) + 1.0)

    out = te.compute(shape, lambda *ix: gelu(x[ix]) * 2.0 + b[ix[-1]], "out")

    def build(g: Graph) -> None:
        xv = g.add_input("X", TensorType(shape, F32))
        g.op(
            "Add",
            g.op("Mul", g.op("Gelu", xv), g.add_constant(np.ones(1, np.float32), "two")),
            g.add_constant(np.zeros(shape[-1], np.float32), "b"),
        )

    def reference(xs: NDArray, bs: NDArray) -> NDArray:
        from math import erf

        return xs * 0.5 * (np.vectorize(erf)(xs * 0.7071067811865476) + 1) * 2 + bs

    rng = np.random.default_rng(seed)
    lowered = LoweredGroup(
        _group("gelu_group", "injective", build),
        te.create_prim_func([x, b, out], "gelu_group"),
        None,
        None,
        {},
    )
    return Workload(lowered, [_rand(rng, *shape), _rand(rng, shape[-1])], reference)
