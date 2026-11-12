"""Tensor expressions: placeholders, compute stages, reductions, and PrimFuncs from them.

A computation is declared as a graph of tensors::

    A = te.placeholder((M, K), DType.float32, "A")
    B = te.placeholder((K, N), DType.float32, "B")
    k = te.reduce_axis(K, "k")
    C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), "C")
    func = te.create_prim_func([A, B, C])

Each compute stage becomes one block with one loop per axis (spatial axes,
then reduction axes) under the function's root block, which allocates every
stage that is not a parameter.
"""

from __future__ import annotations

import itertools
import math
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from tessera.dtype import DType
from tessera.tir.analysis import infer_regions
from tessera.tir.expr import (
    Binary,
    BinaryOp,
    BufferLoad,
    Expr,
    ExprLike,
    FloatImm,
    IntImm,
    Var,
    as_expr,
)
from tessera.tir.functor import ExprVisitor, substitute
from tessera.tir.stmt import (
    TRUE,
    Block,
    BlockRealize,
    Buffer,
    BufferStore,
    For,
    ForKind,
    IterKind,
    IterVar,
    PrimFunc,
    Range,
    Stmt,
    seq,
)

type Combiner = Literal["sum", "max", "min"]
_COMBINE: dict[Combiner, BinaryOp] = {"sum": "add", "max": "max", "min": "min"}

_COUNTER = itertools.count()
_TENSORS: weakref.WeakValueDictionary[int, Tensor] = weakref.WeakValueDictionary()


class Tensor:
    """A tensor of a computation: a placeholder (an input) or the result of a stage.

    Indexing it (``A[i, j]``) gives the load of that element, for use inside
    ``compute`` bodies.
    """

    def __init__(self, shape: Sequence[int], dtype: DType, name: str, op: Operation) -> None:
        self.shape = tuple(int(d) for d in shape)
        self.dtype = dtype
        self.name = name
        self.op = op
        self.buffer = Buffer(name, self.shape, dtype)
        self.order = next(_COUNTER)
        _TENSORS[id(self.buffer)] = self

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __getitem__(self, indices: ExprLike | tuple[ExprLike, ...]) -> BufferLoad:
        items = indices if isinstance(indices, tuple) else (indices,)
        if len(items) != self.ndim:
            raise IndexError(f"{self.name} has rank {self.ndim}, indexed with {len(items)}")
        return BufferLoad(self.buffer, tuple(as_expr(i) for i in items))

    def __repr__(self) -> str:
        return f"Tensor({self.name}, {self.shape}, {self.dtype})"


@dataclass(eq=False)
class PlaceholderOp:
    name: str


@dataclass(eq=False)
class ComputeOp:
    """``out[axis] = body``, or with a combiner the combination of ``body`` over
    ``reduce_axis``."""

    name: str
    axis: tuple[IterVar, ...]
    reduce_axis: tuple[IterVar, ...]
    body: Expr
    combiner: Combiner | None
    init: Expr | None
    inputs: tuple[Tensor, ...]


type Operation = PlaceholderOp | ComputeOp


class Reduce(Expr):
    """A reduction of ``source`` over ``axis`` (only valid as a compute body)."""

    __slots__ = ("axis", "combiner", "init", "source")

    def __init__(
        self, combiner: Combiner, source: Expr, axis: tuple[IterVar, ...], init: Expr
    ) -> None:
        self.combiner = combiner
        self.source = source
        self.axis = axis
        self.init = init

    @property
    def dtype(self) -> DType:
        return self.source.dtype


def placeholder(shape: Sequence[int], dtype: DType = DType.float32, name: str = "X") -> Tensor:
    """An input tensor."""
    return Tensor(shape, dtype, name, PlaceholderOp(name))


def reduce_axis(extent: int, name: str = "k") -> IterVar:
    """A reduction axis ``0 <= name < extent`` for ``sum``/``max``/``min``."""
    return IterVar(Var(name), Range.of(extent), IterKind.REDUCE)


def _axes(axis: IterVar | Sequence[IterVar]) -> tuple[IterVar, ...]:
    out = (axis,) if isinstance(axis, IterVar) else tuple(axis)
    for a in out:
        if a.kind != IterKind.REDUCE:
            raise ValueError(f"{a.var.name} is not a reduction axis")
    return out


def sum(expr: ExprLike, axis: IterVar | Sequence[IterVar]) -> Reduce:
    """The sum of ``expr`` over the reduction axes."""
    e = as_expr(expr, DType.float32)
    return Reduce("sum", e, _axes(axis), as_expr(0, e.dtype))


def max(expr: ExprLike, axis: IterVar | Sequence[IterVar]) -> Reduce:
    """The maximum of ``expr`` over the reduction axes."""
    e = as_expr(expr, DType.float32)
    return Reduce("max", e, _axes(axis), _lowest(e.dtype))


def min(expr: ExprLike, axis: IterVar | Sequence[IterVar]) -> Reduce:
    """The minimum of ``expr`` over the reduction axes."""
    e = as_expr(expr, DType.float32)
    return Reduce("min", e, _axes(axis), _highest(e.dtype))


def _lowest(dtype: DType) -> Expr:
    if dtype.is_float:
        return FloatImm(-math.inf, dtype)
    return IntImm(int(np.iinfo(dtype.numpy).min), dtype)


def _highest(dtype: DType) -> Expr:
    if dtype.is_float:
        return FloatImm(math.inf, dtype)
    return IntImm(int(np.iinfo(dtype.numpy).max), dtype)


class _Loads(ExprVisitor):
    def __init__(self) -> None:
        self.buffers: list[Buffer] = []

    def visit_expr(self, e: Expr) -> None:
        if isinstance(e, Reduce):
            raise ValueError("a reduction must be the whole body of a compute stage")
        super().visit_expr(e)

    def visit_buffer_load(self, e: BufferLoad) -> None:
        if e.buffer not in self.buffers:
            self.buffers.append(e.buffer)
        super().visit_buffer_load(e)


def compute(
    shape: Sequence[int],
    fcompute: Callable[..., ExprLike],
    name: str = "compute",
    axis_names: Sequence[str] | None = None,
) -> Tensor:
    """A stage: ``out[i0, i1, ...] = fcompute(i0, i1, ...)``. The body may be a
    ``sum``/``max``/``min`` reduction."""
    shape = tuple(int(d) for d in shape)
    names = list(axis_names) if axis_names is not None else [f"i{d}" for d in range(len(shape))]
    if len(names) != len(shape):
        raise ValueError(f"{name}: {len(names)} axis names for rank {len(shape)}")
    axis = tuple(IterVar(Var(n), Range.of(e)) for n, e in zip(names, shape, strict=True))
    result = fcompute(*(a.var for a in axis))
    if isinstance(result, Reduce):
        body, combiner, init, raxis = result.source, result.combiner, result.init, result.axis
    else:
        body, combiner, init, raxis = as_expr(result, DType.float32), None, None, ()
    if isinstance(body, Reduce):
        raise ValueError(f"{name}: reductions cannot be nested")
    loads = _Loads()
    loads.visit_expr(body)
    inputs: list[Tensor] = []
    for buf in loads.buffers:
        t = _TENSORS.get(id(buf))
        if t is None or t.buffer is not buf:
            raise ValueError(f"{name} reads {buf.name}, which is not a te tensor")
        inputs.append(t)
    op = ComputeOp(name, axis, raxis, body, combiner, init, tuple(inputs))
    return Tensor(shape, body.dtype, name, op)


def create_prim_func(params: Sequence[Tensor], name: str = "main") -> PrimFunc:
    """The PrimFunc computing ``params`` (inputs first, then outputs): one block per
    stage, in dependency order, under a root block that allocates the stages
    that are not parameters."""
    param_set = {id(t) for t in params}
    stages: list[Tensor] = []
    seen: set[int] = set()

    def visit(t: Tensor) -> None:
        if id(t) in seen:
            return
        seen.add(id(t))
        if isinstance(t.op, PlaceholderOp):
            if id(t) not in param_set:
                raise ValueError(f"placeholder {t.name} is read but not a parameter")
            return
        for i in sorted(t.op.inputs, key=lambda x: x.order):
            visit(i)
        stages.append(t)

    for t in sorted(params, key=lambda x: x.order):
        visit(t)
    used_names: set[str] = set()
    nests: list[Stmt] = []
    allocs: list[Buffer] = []
    for t in stages:
        assert isinstance(t.op, ComputeOp)
        block_name = _unique(t.op.name, used_names)
        nests.append(_stage(t, block_name))
        if id(t) not in param_set:
            allocs.append(t.buffer)
    root = Block("root", (), (), (), seq(nests), None, tuple(allocs))
    return PrimFunc(name, tuple(t.buffer for t in params), BlockRealize((), TRUE, root))


def _unique(name: str, used: set[str]) -> str:
    out, n = name, 1
    while out in used or out == "root":
        out = f"{name}_{n}"
        n += 1
    used.add(out)
    return out


def _stage(t: Tensor, block_name: str) -> Stmt:
    op = t.op
    assert isinstance(op, ComputeOp)
    all_axes = op.axis + op.reduce_axis
    block_vars = {a.var: Var(f"v_{a.var.name}") for a in all_axes}
    loop_vars = [Var(a.var.name) for a in all_axes]
    iter_vars = tuple(IterVar(block_vars[a.var], a.dom, a.kind) for a in all_axes)
    idx = tuple(block_vars[a.var] for a in op.axis)
    body_expr = substitute(op.body, dict(block_vars))
    init: Stmt | None = None
    if op.combiner is None:
        body: Stmt = BufferStore(t.buffer, body_expr, idx)
    else:
        assert op.init is not None
        current = BufferLoad(t.buffer, idx)
        combined = Binary(_COMBINE[op.combiner], current, body_expr)
        body = BufferStore(t.buffer, combined, idx)
        init = BufferStore(t.buffer, op.init, idx)
    block = Block(block_name, iter_vars, (), (), body, init)
    reads, writes = infer_regions(block)
    block = Block(block_name, iter_vars, reads, writes, body, init)
    stmt: Stmt = BlockRealize(tuple(loop_vars), TRUE, block)
    for lv, a in zip(reversed(loop_vars), reversed(all_axes), strict=True):
        stmt = For(lv, a.dom.min, a.dom.extent, ForKind.SERIAL, stmt)
    return stmt
