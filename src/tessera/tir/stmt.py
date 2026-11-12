"""Loop-IR statements, buffers and functions.

The loop IR follows the design of TVM's TensorIR. A ``PrimFunc`` is one
kernel: a loop nest over ``Block``s. A block is a unit of computation with
its own iteration variables (spatial or reduce), the buffer regions it reads
and writes, an optional ``init`` statement for reductions and a body. A
``BlockRealize`` binds the block's iteration variables to expressions of the
surrounding loop variables. Schedule primitives rewrite the loops around
blocks and never change what a block computes, which is what makes them
checkable: every schedule must give the same result as the unscheduled
function (tests run both in the interpreter).

Statements are immutable and compare structurally. ``Buffer`` compares by
identity. A function's parameters are buffers: the group's inputs in order,
then its output.
"""

from __future__ import annotations

import enum
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import prod
from typing import Literal

from tessera.dtype import DType
from tessera.tir.expr import AddressOf, BufferLoad, Expr, ExprLike, IntImm, Var, as_expr

type Scope = Literal["global", "local", "shared"]
type Annotations = tuple[tuple[str, int | float | str], ...]


@dataclass(frozen=True, slots=True, eq=False)
class Buffer:
    """A dense row-major array. ``global``: a kernel parameter or device memory;
    ``local``: registers or the stack of one thread; ``shared``: Metal
    threadgroup memory."""

    name: str
    shape: tuple[int, ...]
    dtype: DType = DType.float32
    scope: Scope = "global"

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        return prod(self.shape)

    @property
    def strides(self) -> tuple[int, ...]:
        """Row-major strides in elements."""
        strides: list[int] = []
        step = 1
        for d in reversed(self.shape):
            strides.append(step)
            step *= d
        return tuple(reversed(strides))

    def __getitem__(self, indices: ExprLike | tuple[ExprLike, ...]) -> BufferLoad:
        items = indices if isinstance(indices, tuple) else (indices,)
        return BufferLoad(self, tuple(as_expr(i) for i in items))

    def address(self, *indices: ExprLike) -> AddressOf:
        return AddressOf(self, tuple(as_expr(i) for i in indices))

    def __repr__(self) -> str:
        return f"Buffer({self.name}, {self.shape}, {self.dtype}, {self.scope})"


@dataclass(frozen=True, slots=True)
class Range:
    min: Expr
    extent: Expr

    @classmethod
    def of(cls, extent: ExprLike, min: ExprLike = 0) -> Range:
        return cls(as_expr(min), as_expr(extent))


class IterKind(enum.StrEnum):
    SPATIAL = "spatial"
    REDUCE = "reduce"


@dataclass(frozen=True, slots=True)
class IterVar:
    var: Var
    dom: Range
    kind: IterKind = IterKind.SPATIAL


class ForKind(enum.StrEnum):
    SERIAL = "serial"
    PARALLEL = "parallel"  # CPU: iterations run on the thread pool
    VECTORIZED = "vectorized"  # CPU: SIMD lanes; Metal: unused
    UNROLLED = "unrolled"
    THREAD_BINDING = "thread_binding"  # Metal: bound to a grid axis (see For.thread_tag)


# Grid axes a THREAD_BINDING loop can be bound to (Metal).
THREAD_TAGS = (
    "threadgroup.x",
    "threadgroup.y",
    "threadgroup.z",
    "thread.x",
    "thread.y",
    "thread.z",
)


class Stmt:
    """Base class of every statement node."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class BufferRegion:
    """The part of a buffer a block touches: one range per dimension."""

    buffer: Buffer
    region: tuple[Range, ...]

    @classmethod
    def full(cls, buffer: Buffer) -> BufferRegion:
        return cls(buffer, tuple(Range.of(d) for d in buffer.shape))


@dataclass(frozen=True, slots=True)
class For(Stmt):
    loop_var: Var
    min: Expr
    extent: Expr
    kind: ForKind
    body: Stmt
    thread_tag: str | None = None
    annotations: Annotations = ()


@dataclass(frozen=True, slots=True)
class Block(Stmt):
    """A unit of computation. Names are unique within a function and are how
    schedules refer to blocks."""

    name: str
    iter_vars: tuple[IterVar, ...]
    reads: tuple[BufferRegion, ...]
    writes: tuple[BufferRegion, ...]
    body: Stmt
    init: Stmt | None = None
    alloc_buffers: tuple[Buffer, ...] = ()
    annotations: Annotations = ()


@dataclass(frozen=True, slots=True)
class BlockRealize(Stmt):
    """An instance of ``block`` with its iteration variables bound to ``iter_values``,
    executed only where ``predicate`` holds (imperfect splits)."""

    iter_values: tuple[Expr, ...]
    predicate: Expr
    block: Block


@dataclass(frozen=True, slots=True)
class BufferStore(Stmt):
    buffer: Buffer
    value: Expr
    indices: tuple[Expr, ...]


@dataclass(frozen=True, slots=True)
class SeqStmt(Stmt):
    seq: tuple[Stmt, ...]


@dataclass(frozen=True, slots=True)
class IfThenElse(Stmt):
    condition: Expr
    then_case: Stmt
    else_case: Stmt | None = None


@dataclass(frozen=True, slots=True)
class LetStmt(Stmt):
    var: Var
    value: Expr
    body: Stmt


@dataclass(frozen=True, slots=True)
class Evaluate(Stmt):
    """Evaluate an expression for its effect: a backend intrinsic call."""

    value: Expr


TRUE = IntImm(1, DType.bool_)


@dataclass(frozen=True)
class PrimFunc:
    """One kernel. ``params`` are its global buffers, inputs first, then outputs;
    the compiled kernel takes one pointer per parameter in this order."""

    name: str
    params: tuple[Buffer, ...]
    body: Stmt
    attrs: Annotations = field(default=())

    def attr(self, key: str, default: int | float | str | None = None) -> int | float | str | None:
        return dict(self.attrs).get(key, default)


def seq(stmts: Sequence[Stmt]) -> Stmt:
    """A SeqStmt, flattened; a single statement is returned as is."""
    flat: list[Stmt] = []
    for s in stmts:
        flat.extend(s.seq if isinstance(s, SeqStmt) else (s,))
    return flat[0] if len(flat) == 1 else SeqStmt(tuple(flat))
