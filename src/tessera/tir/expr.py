"""Loop-IR expressions.

Expressions are immutable trees. They compare structurally (two ``Add`` nodes
with equal operands are equal), except ``Var`` and ``Buffer``, which compare
by identity: two loop variables named ``i`` are different variables.

Arithmetic operators build trees (``i * 4 + j``); Python ints and floats are
converted to immediates of the other operand's type. Comparisons have no
operator overloads, because ``==`` must stay structural equality: use
``lt``, ``le``, ``eq``, ``ne``, ``gt``, ``ge``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from tessera.dtype import INDEX, DType

if TYPE_CHECKING:
    from tessera.tir.stmt import Buffer

type BinaryOp = Literal["add", "sub", "mul", "div", "floordiv", "floormod", "min", "max", "pow"]
type CompareOp = Literal["lt", "le", "eq", "ne", "gt", "ge"]
type LogicalOp = Literal["and", "or"]
type ExprLike = Expr | int | float | bool

# Math intrinsics every backend implements for float32 (Call.name).
MATH_INTRINSICS = frozenset(
    {"exp", "log", "sqrt", "rsqrt", "tanh", "erf", "sigmoid", "abs", "floor", "ceil", "fma"}
)


class Expr:
    """Base class of every expression node."""

    __slots__ = ()

    @property
    def dtype(self) -> DType:
        raise NotImplementedError

    def __add__(self, other: ExprLike) -> Expr:
        return Binary("add", self, as_expr(other, self.dtype))

    def __radd__(self, other: ExprLike) -> Expr:
        return Binary("add", as_expr(other, self.dtype), self)

    def __sub__(self, other: ExprLike) -> Expr:
        return Binary("sub", self, as_expr(other, self.dtype))

    def __rsub__(self, other: ExprLike) -> Expr:
        return Binary("sub", as_expr(other, self.dtype), self)

    def __mul__(self, other: ExprLike) -> Expr:
        return Binary("mul", self, as_expr(other, self.dtype))

    def __rmul__(self, other: ExprLike) -> Expr:
        return Binary("mul", as_expr(other, self.dtype), self)

    def __truediv__(self, other: ExprLike) -> Expr:
        return Binary("div", self, as_expr(other, self.dtype))

    def __rtruediv__(self, other: ExprLike) -> Expr:
        return Binary("div", as_expr(other, self.dtype), self)

    def __floordiv__(self, other: ExprLike) -> Expr:
        return Binary("floordiv", self, as_expr(other, self.dtype))

    def __mod__(self, other: ExprLike) -> Expr:
        return Binary("floormod", self, as_expr(other, self.dtype))

    def __neg__(self) -> Expr:
        return Binary("sub", as_expr(0, self.dtype), self)


@dataclass(frozen=True, slots=True)
class IntImm(Expr):
    value: int
    type: DType = INDEX

    @property
    def dtype(self) -> DType:
        return self.type


@dataclass(frozen=True, slots=True)
class FloatImm(Expr):
    value: float
    type: DType = DType.float32

    @property
    def dtype(self) -> DType:
        return self.type


@dataclass(frozen=True, slots=True, eq=False)
class Var(Expr):
    """A variable (loop variable, block iteration variable). Compares by identity."""

    name: str
    type: DType = INDEX

    @property
    def dtype(self) -> DType:
        return self.type

    def __repr__(self) -> str:
        return f"Var({self.name})"


@dataclass(frozen=True, slots=True)
class Binary(Expr):
    """Arithmetic. ``div`` is true division (floats); ``floordiv``/``floormod`` round
    toward negative infinity like Python's ``//`` and ``%``."""

    op: BinaryOp
    a: Expr
    b: Expr

    @property
    def dtype(self) -> DType:
        return self.a.dtype


@dataclass(frozen=True, slots=True)
class Compare(Expr):
    op: CompareOp
    a: Expr
    b: Expr

    @property
    def dtype(self) -> DType:
        return DType.bool_


@dataclass(frozen=True, slots=True)
class Logical(Expr):
    op: LogicalOp
    a: Expr
    b: Expr

    @property
    def dtype(self) -> DType:
        return DType.bool_


@dataclass(frozen=True, slots=True)
class Not(Expr):
    a: Expr

    @property
    def dtype(self) -> DType:
        return DType.bool_


@dataclass(frozen=True, slots=True)
class Select(Expr):
    """``cond ? true_value : false_value``; both sides may be evaluated."""

    cond: Expr
    true_value: Expr
    false_value: Expr

    @property
    def dtype(self) -> DType:
        return self.true_value.dtype


@dataclass(frozen=True, slots=True)
class Cast(Expr):
    type: DType
    value: Expr

    @property
    def dtype(self) -> DType:
        return self.type


@dataclass(frozen=True, slots=True)
class Call(Expr):
    """A pure intrinsic: a math function (``MATH_INTRINSICS``) or a backend
    intrinsic named ``<backend>.<name>`` (microkernels, barriers), used inside
    ``Evaluate``."""

    name: str
    args: tuple[Expr, ...]
    type: DType = DType.float32

    @property
    def dtype(self) -> DType:
        return self.type


@dataclass(frozen=True, slots=True)
class BufferLoad(Expr):
    buffer: Buffer
    indices: tuple[Expr, ...]

    @property
    def dtype(self) -> DType:
        return self.buffer.dtype


@dataclass(frozen=True, slots=True)
class AddressOf(Expr):
    """A pointer to ``buffer[indices]``, for passing tiles to microkernels."""

    buffer: Buffer
    indices: tuple[Expr, ...]

    @property
    def dtype(self) -> DType:
        return INDEX


def as_expr(value: ExprLike, dtype: DType = INDEX) -> Expr:
    """Wrap a Python number as an immediate of ``dtype`` (expressions pass through)."""
    if isinstance(value, Expr):
        return value
    if isinstance(value, bool):
        return IntImm(int(value), DType.bool_)
    if isinstance(value, int):
        return FloatImm(float(value), dtype) if dtype.is_float else IntImm(value, dtype)
    if isinstance(value, float):
        return FloatImm(value, dtype if dtype.is_float else DType.float32)
    raise TypeError(f"cannot make an expression from {value!r}")


def const(value: int | float, dtype: DType = INDEX) -> Expr:
    return as_expr(value, dtype)


def lt(a: ExprLike, b: ExprLike) -> Compare:
    return _compare("lt", a, b)


def le(a: ExprLike, b: ExprLike) -> Compare:
    return _compare("le", a, b)


def eq(a: ExprLike, b: ExprLike) -> Compare:
    return _compare("eq", a, b)


def ne(a: ExprLike, b: ExprLike) -> Compare:
    return _compare("ne", a, b)


def gt(a: ExprLike, b: ExprLike) -> Compare:
    return _compare("gt", a, b)


def ge(a: ExprLike, b: ExprLike) -> Compare:
    return _compare("ge", a, b)


def logical_and(a: Expr, b: Expr) -> Logical:
    return Logical("and", a, b)


def logical_or(a: Expr, b: Expr) -> Logical:
    return Logical("or", a, b)


def select(cond: Expr, true_value: ExprLike, false_value: ExprLike) -> Select:
    t = as_expr(true_value, false_value.dtype if isinstance(false_value, Expr) else DType.float32)
    return Select(cond, t, as_expr(false_value, t.dtype))


def minimum(a: ExprLike, b: ExprLike) -> Binary:
    x, y = _pair(a, b)
    return Binary("min", x, y)


def maximum(a: ExprLike, b: ExprLike) -> Binary:
    x, y = _pair(a, b)
    return Binary("max", x, y)


def call(name: str, *args: ExprLike, dtype: DType = DType.float32) -> Call:
    return Call(name, tuple(as_expr(a, dtype) for a in args), dtype)


def _pair(a: ExprLike, b: ExprLike) -> tuple[Expr, Expr]:
    if isinstance(a, Expr):
        return a, as_expr(b, a.dtype)
    if isinstance(b, Expr):
        return as_expr(a, b.dtype), b
    return as_expr(a), as_expr(b)


def _compare(op: CompareOp, a: ExprLike, b: ExprLike) -> Compare:
    x, y = _pair(a, b)
    return Compare(op, x, y)
