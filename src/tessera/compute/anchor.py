"""Compute definitions of the anchors and reductions: Conv2d, MatMul, pooling,
ReduceSum/Mean/Max, Softmax and LayerNorm.

Each one builds its stages with ``te.compute`` and returns an ``AnchorResult``:
the stage whose block is the group's main block, the tensor the epilogue
reads (the main stage itself, or for ReduceMean and AvgPool2d the division of
its sums, which then belongs to the epilogue), the problem sizes a template
needs (``dims``) and the statistics stages of Softmax and LayerNorm. Every
block is annotated with its ``role``: ``main``, ``max``, ``sum``, ``mean`` or
``var``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import prod

from tessera import te
from tessera.compute.elementwise import c
from tessera.compute.tensor import Lazy, TensorLike, broadcast_index, clamp
from tessera.ir.graph import Node
from tessera.tir.arith import Analyzer, Interval
from tessera.tir.expr import Cast, Expr, IntImm, Select, Var, call, ge, logical_and, lt
from tessera.tir.expr import maximum as emax
from tessera.tir.expr import minimum as emin
from tessera.tir.stmt import IterVar

ROLE = "role"


@dataclass
class AnchorResult:
    main: te.Tensor
    value: TensorLike
    dims: dict[str, int]
    stages: list[te.Tensor] = field(default_factory=list)


type AnchorDefinition = Callable[[Node, list[TensorLike]], AnchorResult]

ANCHORS: dict[str, AnchorDefinition] = {}


def _register(*names: str) -> Callable[[AnchorDefinition], AnchorDefinition]:
    def wrap(fn: AnchorDefinition) -> AnchorDefinition:
        for n in names:
            ANCHORS[n] = fn
        return fn

    return wrap


def guarded_load(
    t: TensorLike, indices: Sequence[Expr], ranges: dict[Var, Interval], fill: Expr
) -> Expr:
    """``t[indices]`` where in bounds and ``fill`` elsewhere (padding). Dimensions that
    provably stay in bounds get no test; the others clamp the index, because both
    sides of a select may be evaluated."""
    an = Analyzer(ranges)
    conds: list[Expr] = []
    idx: list[Expr] = []
    for e, n in zip(indices, t.shape, strict=True):
        b = an.bound(e)
        low = b.lo is None or b.lo < 0
        high = b.hi is None or b.hi >= n
        if low:
            conds.append(ge(e, 0))
        if high:
            conds.append(lt(e, n))
        idx.append(clamp(e, 0, n - 1) if low or high else e)
    value = t[tuple(idx)]
    if not conds:
        return value
    cond = conds[0]
    for x in conds[1:]:
        cond = logical_and(cond, x)
    return Select(cond, value, fill)


def _ranges(pairs: Sequence[tuple[Var | IterVar, int]]) -> dict[Var, Interval]:
    return {(v.var if isinstance(v, IterVar) else v): Interval(0, n - 1) for v, n in pairs}


# -- Conv2d -------------------------------------------------------------------------------


@_register("Conv2d")
def conv2d(node: Node, xs: list[TensorLike]) -> AnchorResult:
    x, w = xs[0], xs[1]
    bias = xs[2] if len(xs) == 3 else None
    nhwc = node.str_attr("layout") == "NHWC"
    sh, sw = node.ints_attr("strides")
    dh, dw = node.ints_attr("dilations")
    pt, pl, pb, pr = node.ints_attr("pads")
    groups = node.int_attr("groups")
    if nhwc:
        n, h, wd, ch = x.shape
        kh_n, kw_n, i_n, o_n = w.shape
    else:
        n, ch, h, wd = x.shape
        o_n, i_n, kh_n, kw_n = w.shape
    out_shape = node.output.shape
    oh_n, ow_n = (out_shape[1], out_shape[2]) if nhwc else (out_shape[2], out_shape[3])
    per_group = o_n // groups
    kh, kw = te.reduce_axis(kh_n, "kh"), te.reduce_axis(kw_n, "kw")
    ic = te.reduce_axis(i_n, "ic") if i_n > 1 else None
    axes = [kh, kw] if ic is None else [kh, kw, ic]
    dtype = x.dtype

    def split(out: tuple[Var, ...]) -> tuple[Var, Var, Var, Var]:
        b, a1, a2, a3 = out
        return (b, a1, a2, a3) if nhwc else (b, a2, a3, a1)

    def body(*out: Var) -> te.Reduce:
        b, oh, ow, o = split(out)
        ic_e: Expr = IntImm(0) if ic is None else ic.var
        channel = ic_e if groups == 1 else (o // per_group) * i_n + ic_e
        hh = oh * sh + kh * dh - pt
        ww = ow * sw + kw * dw - pl
        ranges = _ranges([(b, n), (oh, oh_n), (ow, ow_n), (o, o_n), (kh, kh_n), (kw, kw_n)])
        if ic is not None:
            ranges[ic.var] = Interval(0, i_n - 1)
        if nhwc:
            xv = guarded_load(x, (b, hh, ww, channel), ranges, c(0, dtype))
            wv = w[(kh.var, kw.var, ic_e, o)]
        else:
            xv = guarded_load(x, (b, channel, hh, ww), ranges, c(0, dtype))
            wv = w[(o, ic_e, kh.var, kw.var)]
        return te.sum(xv * wv, axis=axes)

    main = te.compute(
        out_shape,
        body,
        node.name,
        axis_names=["n", "oh", "ow", "o"] if nhwc else ["n", "o", "oh", "ow"],
        annotations={ROLE: "main"},
        init=None if bias is None else (lambda *out: bias[(split(out)[3],)]),
    )
    dims = {
        "N": n, "H": h, "W": wd, "C": ch, "KH": kh_n, "KW": kw_n, "O": o_n,
        "OH": oh_n, "OW": ow_n, "SH": sh, "SW": sw, "DH": dh, "DW": dw, "G": groups,
        "PT": pt, "PL": pl, "PB": pb, "PR": pr,
    }  # fmt: skip
    return AnchorResult(main, main, dims)


# -- MatMul -------------------------------------------------------------------------------


@_register("MatMul")
def matmul(node: Node, xs: list[TensorLike]) -> AnchorResult:
    a, b = xs
    out_shape = node.output.shape
    batch = out_shape[:-2]
    m, n = out_shape[-2:]
    k_n = a.shape[-1]
    k = te.reduce_axis(k_n, "k")

    def body(*out: Var) -> te.Reduce:
        bi, i, j = out[:-2], out[-2], out[-1]
        ai = broadcast_index(batch, a.shape[:-2], bi)
        bj = broadcast_index(batch, b.shape[:-2], bi)
        return te.sum(a[(*ai, i, k.var)] * b[(*bj, k.var, j)], axis=k)

    names = [f"b{d}" for d in range(len(batch))] + ["m", "n"]
    main = te.compute(out_shape, body, node.name, axis_names=names, annotations={ROLE: "main"})
    return AnchorResult(main, main, {"B": prod(batch), "M": m, "N": n, "K": k_n})


# -- pooling -------------------------------------------------------------------------------


@_register("MaxPool2d", "AvgPool2d")
def pool2d(node: Node, xs: list[TensorLike]) -> AnchorResult:
    (x,) = xs
    nhwc = node.str_attr("layout") == "NHWC"
    kh_n, kw_n = node.ints_attr("kernel")
    sh, sw = node.ints_attr("strides")
    pt, pl, _, _ = node.ints_attr("pads")
    is_max = node.op == "MaxPool2d"
    if nhwc:
        n, h, wd, ch = x.shape
    else:
        n, ch, h, wd = x.shape
    out_shape = node.output.shape
    oh_n, ow_n = (out_shape[1], out_shape[2]) if nhwc else (out_shape[2], out_shape[3])
    kh, kw = te.reduce_axis(kh_n, "kh"), te.reduce_axis(kw_n, "kw")
    dtype = x.dtype
    fill = c(float("-inf"), dtype) if is_max else c(0, dtype)

    def split(out: tuple[Var, ...]) -> tuple[Var, Var, Var, Var]:
        b, a1, a2, a3 = out
        return (b, a1, a2, a3) if nhwc else (b, a2, a3, a1)

    def body(*out: Var) -> te.Reduce:
        b, oh, ow, ci = split(out)
        hh, ww = oh * sh + kh - pt, ow * sw + kw - pl
        ranges = _ranges([(b, n), (oh, oh_n), (ow, ow_n), (ci, ch), (kh, kh_n), (kw, kw_n)])
        idx = (b, hh, ww, ci) if nhwc else (b, ci, hh, ww)
        value = guarded_load(x, idx, ranges, fill)
        return te.max(value, axis=[kh, kw]) if is_max else te.sum(value, axis=[kh, kw])

    names = ["n", "oh", "ow", "c"] if nhwc else ["n", "c", "oh", "ow"]
    main = te.compute(out_shape, body, node.name, axis_names=names, annotations={ROLE: "main"})
    dims = {"N": n, "H": h, "W": wd, "C": ch, "KH": kh_n, "KW": kw_n,
            "OH": oh_n, "OW": ow_n, "SH": sh, "SW": sw}  # fmt: skip
    if is_max:
        return AnchorResult(main, main, dims)
    if node.int_attr("count_include_pad"):
        value: TensorLike = Lazy(out_shape, dtype, lambda i: main[i] / c(kh_n * kw_n, dtype))
        return AnchorResult(main, value, dims)

    def count(start: Expr, k: int, size: int) -> Expr:
        return emin(start + k, size) - emax(start, 0)

    def average(i: tuple[Expr, ...]) -> Expr:
        _, oh, ow, _ = (i[0], i[1], i[2], i[3]) if nhwc else (i[0], i[2], i[3], i[1])
        valid = count(oh * sh - pt, kh_n, h) * count(ow * sw - pl, kw_n, wd)
        return main[i] / Cast(dtype, valid)

    return AnchorResult(main, Lazy(out_shape, dtype, average), dims)


# -- ReduceSum, ReduceMean, ReduceMax ---------------------------------------------------------


def _outer_inner(shape: Sequence[int], axes: Sequence[int]) -> tuple[int, int, int]:
    last = max(axes)
    outer = prod(d for i, d in enumerate(shape) if i < last and i not in axes)
    inner = prod(shape[last + 1 :])
    return outer, prod(shape[a] for a in axes), inner


@_register("ReduceSum", "ReduceMean", "ReduceMax")
def reduce(node: Node, xs: list[TensorLike]) -> AnchorResult:
    (x,) = xs
    axes = node.ints_attr("axes")
    keep = bool(node.int_attr("keepdims"))
    out_shape = node.output.shape
    raxes = {a: te.reduce_axis(x.shape[a], f"r{a}") for a in axes}
    kept = [d for d in range(len(x.shape)) if d not in axes]

    def body(*out: Var) -> te.Reduce:
        idx: list[Expr] = []
        for d in range(len(x.shape)):
            if d in raxes:
                idx.append(raxes[d].var)
            else:
                idx.append(out[d] if keep else out[kept.index(d)])
        value = x[tuple(idx)]
        if node.op == "ReduceMax":
            return te.max(value, axis=list(raxes.values()))
        return te.sum(value, axis=list(raxes.values()))

    names = [f"i{d}" for d in range(len(out_shape))]
    main = te.compute(out_shape, body, node.name, axis_names=names, annotations={ROLE: "main"})
    outer, r, inner = _outer_inner(x.shape, axes)
    dims = {"OUTER": outer, "R": r, "INNER": inner}
    if node.op != "ReduceMean":
        return AnchorResult(main, main, dims)
    mean = Lazy(out_shape, x.dtype, lambda i: main[i] / c(r, x.dtype))
    return AnchorResult(main, mean, dims)


# -- Softmax ---------------------------------------------------------------------------------


@_register("Softmax")
def softmax(node: Node, xs: list[TensorLike]) -> AnchorResult:
    (x,) = xs
    axis = node.int_attr("axis")
    shape = x.shape
    stat_shape = shape[:axis] + shape[axis + 1 :]
    names = [f"i{d}" for d in range(len(shape))]
    stat_names = names[:axis] + names[axis + 1 :]
    r1, r2 = te.reduce_axis(shape[axis], "r"), te.reduce_axis(shape[axis], "r")

    def at(stat: tuple[Expr, ...], r: Expr) -> tuple[Expr, ...]:
        return (*stat[:axis], r, *stat[axis:])

    mx = te.compute(
        stat_shape,
        lambda *s: te.max(x[at(s, r1.var)], axis=r1),
        f"{node.name}_max",
        axis_names=stat_names,
        annotations={ROLE: "max"},
    )
    sm = te.compute(
        stat_shape,
        lambda *s: te.sum(call("exp", x[at(s, r2.var)] - mx[s]), axis=r2),
        f"{node.name}_sum",
        axis_names=stat_names,
        annotations={ROLE: "sum"},
    )

    def body(*i: Var) -> Expr:
        stat = i[:axis] + i[axis + 1 :]
        return call("exp", x[i] - mx[stat]) / sm[stat]

    main = te.compute(shape, body, node.name, axis_names=names, annotations={ROLE: "main"})
    dims = {"OUTER": prod(shape[:axis]), "R": shape[axis], "INNER": prod(shape[axis + 1 :])}
    return AnchorResult(main, main, dims, [mx, sm])


# -- LayerNorm -------------------------------------------------------------------------------


@_register("LayerNorm")
def layernorm(node: Node, xs: list[TensorLike]) -> AnchorResult:
    x, scale, bias = xs
    axis = node.int_attr("axis")
    eps = node.float_attr("epsilon")
    shape = x.shape
    outer_shape, norm_shape = shape[:axis], shape[axis:]
    count = prod(norm_shape)
    dtype = x.dtype
    names = [f"i{d}" for d in range(len(shape))]
    r1 = [te.reduce_axis(n, f"r{axis + d}") for d, n in enumerate(norm_shape)]
    r2 = [te.reduce_axis(n, f"r{axis + d}") for d, n in enumerate(norm_shape)]

    total = te.compute(
        outer_shape,
        lambda *o: te.sum(x[(*o, *(r.var for r in r1))], axis=r1),
        f"{node.name}_mean",
        axis_names=names[:axis],
        annotations={ROLE: "mean"},
    )

    def mean(o: tuple[Expr, ...]) -> Expr:
        return total[o] / c(count, dtype)

    def sq(*o: Var) -> te.Reduce:
        centered = x[(*o, *(r.var for r in r2))] - mean(o)
        return te.sum(centered * centered, axis=r2)

    var = te.compute(
        outer_shape, sq, f"{node.name}_var", axis_names=names[:axis], annotations={ROLE: "var"}
    )

    def body(*i: Var) -> Expr:
        o, inner = i[:axis], i[axis:]
        centered = x[i] - mean(o)
        std = call("sqrt", var[o] / c(count, dtype) + c(eps, dtype))
        return centered / std * scale[inner] + bias[inner]

    main = te.compute(shape, body, node.name, axis_names=names, annotations={ROLE: "main"})
    dims = {"OUTER": prod(outer_shape), "R": count, "INNER": 1}
    return AnchorResult(main, main, dims, [total, var])
