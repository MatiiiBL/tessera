"""Compute definitions of the elementwise, broadcast and injective operators.

Every definition returns a ``Lazy`` tensor: these operators never get a block
of their own inside a fused group, they are inlined into the loads of the
anchor (prologue) or into the epilogue's single store.

Loads guarded by a ``select`` (padding, concatenation) still clamp their
indices into the buffer, because both sides of a select may be evaluated.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from tessera.compute.tensor import Lazy, TensorLike, broadcast, broadcast_index, clamp
from tessera.dtype import INDEX, DType
from tessera.ir.graph import Node
from tessera.tir.expr import (
    Binary,
    Cast,
    Compare,
    Expr,
    FloatImm,
    IntImm,
    Select,
    call,
    ge,
    logical_and,
    lt,
    maximum,
    minimum,
)

type Definition = Callable[[Node, list[TensorLike]], TensorLike]

DEFINITIONS: dict[str, Definition] = {}


def _register(*names: str) -> Callable[[Definition], Definition]:
    def wrap(fn: Definition) -> Definition:
        for n in names:
            DEFINITIONS[n] = fn
        return fn

    return wrap


def c(value: float, dtype: DType) -> Expr:
    """A constant of ``dtype``."""
    if dtype.is_float:
        return FloatImm(float(value), dtype)
    if dtype == DType.bool_:
        return IntImm(int(bool(value)), DType.bool_)
    return IntImm(int(value), dtype)


def _f(name: str, x: Expr) -> Expr:
    return call(name, x, dtype=x.dtype)


def _sigmoid(x: Expr) -> Expr:
    return _f("sigmoid", x)


def _clip(x: Expr, lo: float, hi: float) -> Expr:
    """``min(max(x, lo), hi)`` as the reference computes it (bounds outside an integer
    type's range are dropped; integer bounds round inward)."""
    dt = x.dtype
    if dt.is_float:
        out = x if lo == -math.inf else maximum(x, c(lo, dt))
        return out if hi == math.inf else minimum(out, c(hi, dt))
    info = np.iinfo(dt.numpy) if dt != DType.bool_ else None
    if info is not None:
        lo, hi = max(lo, info.min), min(hi, info.max)
    out = x if not math.isfinite(lo) else maximum(x, c(math.ceil(lo), dt))
    return out if not math.isfinite(hi) else minimum(out, c(math.floor(hi), dt))


def _gelu(x: Expr, approximate: str) -> Expr:
    dt = x.dtype
    half = c(0.5, dt)
    if approximate == "tanh":
        inner = c(math.sqrt(2 / math.pi), dt) * (x + c(0.044715, dt) * x * x * x)
        return half * x * (c(1.0, dt) + _f("tanh", inner))
    return half * x * (c(1.0, dt) + _f("erf", x * c(1 / math.sqrt(2), dt)))


_UNARY: dict[str, Callable[[Node, Expr], Expr]] = {
    "Relu": lambda n, x: maximum(x, c(0, x.dtype)),
    "Sigmoid": lambda n, x: _sigmoid(x),
    "Tanh": lambda n, x: _f("tanh", x),
    "Exp": lambda n, x: _f("exp", x),
    "Log": lambda n, x: _f("log", x),
    "Sqrt": lambda n, x: _f("sqrt", x),
    "Rsqrt": lambda n, x: c(1.0, x.dtype) / _f("sqrt", x),
    "Erf": lambda n, x: _f("erf", x),
    "Neg": lambda n, x: Binary("sub", c(0, x.dtype), x),
    "Abs": lambda n, x: (
        _f("abs", x) if x.dtype.is_float else Select(lt(x, c(0, x.dtype)), c(0, x.dtype) - x, x)
    ),
    "Silu": lambda n, x: x * _sigmoid(x),
    "HardSwish": lambda n, x: x * _clip(x * c(1 / 6, x.dtype) + c(0.5, x.dtype), 0.0, 1.0),
    "HardSigmoid": lambda n, x: _clip(
        c(n.float_attr("alpha"), x.dtype) * x + c(n.float_attr("beta"), x.dtype), 0.0, 1.0
    ),
    "Gelu": lambda n, x: _gelu(x, n.str_attr("approximate")),
    "Clip": lambda n, x: _clip(x, n.float_attr("min"), n.float_attr("max")),
}


def _unary(node: Node, xs: list[TensorLike]) -> TensorLike:
    (x,) = xs
    fn = _UNARY[node.op]
    return Lazy(x.shape, x.dtype, lambda i: fn(node, x[i]))


for _name in _UNARY:
    DEFINITIONS[_name] = _unary


@_register("Cast")
def _cast(node: Node, xs: list[TensorLike]) -> TensorLike:
    (x,) = xs
    to = DType(node.str_attr("to"))
    if to == x.dtype:
        return x
    if to == DType.bool_:
        return Lazy(x.shape, to, lambda i: Compare("ne", x[i], c(0, x.dtype)))
    return Lazy(x.shape, to, lambda i: Cast(to, x[i]))


_ARITH = {"Add": "add", "Sub": "sub", "Mul": "mul", "Div": "div", "Pow": "pow",
          "Maximum": "max", "Minimum": "min"}  # fmt: skip
_COMPARE = {"Equal": "eq", "Less": "lt", "Greater": "gt"}


@_register(*_ARITH, *_COMPARE)
def _binary(node: Node, xs: list[TensorLike]) -> TensorLike:
    a, b = xs
    shape = node.output.shape

    def element(i: tuple[Expr, ...]) -> Expr:
        x, y = broadcast(a, shape, i), broadcast(b, shape, i)
        if node.op in _COMPARE:
            return Compare(_COMPARE[node.op], x, y)  # type: ignore[arg-type]
        return Binary(_ARITH[node.op], x, y)  # type: ignore[arg-type]

    dtype = DType.bool_ if node.op in _COMPARE else a.dtype
    return Lazy(shape, dtype, element)


@_register("Where")
def _where(node: Node, xs: list[TensorLike]) -> TensorLike:
    cond, a, b = xs
    shape = node.output.shape
    return Lazy(
        shape,
        a.dtype,
        lambda i: Select(broadcast(cond, shape, i), broadcast(a, shape, i), broadcast(b, shape, i)),
    )


@_register("BatchNorm")
def _batchnorm(node: Node, xs: list[TensorLike]) -> TensorLike:
    x, scale, bias, mean, var = xs
    eps = c(node.float_attr("epsilon"), x.dtype)
    one = c(1.0, x.dtype)

    def element(i: tuple[Expr, ...]) -> Expr:
        ch = (i[1],)
        inv = one / _f("sqrt", var[ch] + eps)
        return (x[i] - mean[ch]) * inv * scale[ch] + bias[ch]

    return Lazy(x.shape, x.dtype, element)


# -- injective -------------------------------------------------------------------------


def row_major(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides, step = [], 1
    for d in reversed(shape):
        strides.append(step)
        step *= d
    return tuple(reversed(strides))


@_register("Reshape")
def _reshape(node: Node, xs: list[TensorLike]) -> TensorLike:
    (x,) = xs
    out_shape = node.output.shape

    def element(i: tuple[Expr, ...]) -> Expr:
        flat: Expr = IntImm(0)
        for e, s in zip(i, row_major(out_shape), strict=True):
            flat = flat + e * s
        idx = tuple((flat // s) % d for s, d in zip(row_major(x.shape), x.shape, strict=True))
        return x[idx]

    return Lazy(out_shape, x.dtype, element)


@_register("Transpose")
def _transpose(node: Node, xs: list[TensorLike]) -> TensorLike:
    (x,) = xs
    perm = node.ints_attr("perm")

    def element(i: tuple[Expr, ...]) -> Expr:
        idx: list[Expr] = [IntImm(0)] * len(perm)
        for out_axis, in_axis in enumerate(perm):
            idx[in_axis] = i[out_axis]
        return x[tuple(idx)]

    return Lazy(node.output.shape, x.dtype, element)


@_register("Slice")
def _slice(node: Node, xs: list[TensorLike]) -> TensorLike:
    (x,) = xs
    starts, steps = node.ints_attr("starts"), node.ints_attr("steps")
    return Lazy(
        node.output.shape,
        x.dtype,
        lambda i: x[tuple(e * st + s for e, s, st in zip(i, starts, steps, strict=True))],
    )


@_register("Concat")
def _concat(node: Node, xs: list[TensorLike]) -> TensorLike:
    axis = node.int_attr("axis")
    sizes = [t.shape[axis] for t in xs]

    def element(i: tuple[Expr, ...]) -> Expr:
        pos = i[axis]
        out: Expr | None = None
        start = sum(sizes)
        for t, n in zip(reversed(xs), reversed(sizes), strict=True):
            start -= n
            local = clamp(pos - start, 0, n - 1)
            value = t[(*i[:axis], local, *i[axis + 1 :])]
            out = value if out is None else Select(lt(pos, start + n), value, out)
        assert out is not None
        return out

    return Lazy(node.output.shape, xs[0].dtype, element)


@_register("Gather")
def _gather(node: Node, xs: list[TensorLike]) -> TensorLike:
    data, indices = xs
    axis = node.int_attr("axis")
    q = len(indices.shape)
    n = data.shape[axis]

    def element(i: tuple[Expr, ...]) -> Expr:
        raw = indices[tuple(i[axis : axis + q])]
        idx = raw if raw.dtype == INDEX else Cast(INDEX, raw)
        wrapped = Select(lt(idx, 0), idx + n, idx)
        return data[(*i[:axis], clamp(wrapped, 0, n - 1), *i[axis + q :])]

    return Lazy(node.output.shape, data.dtype, element)


@_register("Expand")
def _expand(node: Node, xs: list[TensorLike]) -> TensorLike:
    (x,) = xs
    shape = node.output.shape
    return Lazy(shape, x.dtype, lambda i: x[broadcast_index(shape, x.shape, i)])


@_register("Pad")
def _pad(node: Node, xs: list[TensorLike]) -> TensorLike:
    (x,) = xs
    pads = node.ints_attr("pads")
    rank = len(x.shape)
    value = c(node.float_attr("value"), x.dtype)
    if rank == 0:
        return x

    def element(i: tuple[Expr, ...]) -> Expr:
        inside: Expr | None = None
        idx: list[Expr] = []
        for d, e in enumerate(i):
            before, n = pads[d], x.shape[d]
            if before == 0 and pads[d + rank] == 0:
                idx.append(e)
                continue
            src = e - before
            cond = logical_and(ge(src, 0), lt(src, n))
            inside = cond if inside is None else logical_and(inside, cond)
            idx.append(clamp(src, 0, n - 1))
        if inside is None:
            return x[tuple(idx)]
        return Select(inside, x[tuple(idx)], value)

    return Lazy(node.output.shape, x.dtype, element)
