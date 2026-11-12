"""Tensors as functions of their indices, for composing operators before lowering.

A compute definition receives its inputs as ``TensorLike`` values: anything
with a shape, an element type and ``__getitem__`` from index expressions to
an element expression. Placeholders and stages (``te.Tensor``) are
``TensorLike``, and so is ``Lazy``, which computes each element on demand
from other tensors. Elementwise, broadcast and injective operators return
``Lazy`` tensors, so chaining them composes their expressions: when a
reduction or an anchor reads a ``Lazy`` input, the prologue ends up inlined
in its loads, and an epilogue chain becomes one expression stored by one
block.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

from tessera.dtype import DType
from tessera.tir.expr import Expr, IntImm, maximum, minimum


class TensorLike(Protocol):
    """A tensor whose elements are expressions of their indices."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> DType: ...

    def __getitem__(self, indices: tuple[Expr, ...]) -> Expr: ...


class Lazy:
    """A tensor computed element by element by ``fn`` (indices -> element)."""

    def __init__(
        self, shape: Sequence[int], dtype: DType, fn: Callable[[tuple[Expr, ...]], Expr]
    ) -> None:
        self._shape = tuple(int(d) for d in shape)
        self._dtype = dtype
        self.fn = fn

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> DType:
        return self._dtype

    def __getitem__(self, indices: tuple[Expr, ...]) -> Expr:
        if len(indices) != len(self._shape):
            raise IndexError(f"rank {len(self._shape)} tensor indexed with {len(indices)}")
        return self.fn(indices)


def broadcast_index(
    out_shape: Sequence[int], shape: Sequence[int], indices: Sequence[Expr]
) -> tuple[Expr, ...]:
    """The index into a tensor of ``shape`` broadcast to ``out_shape`` (NumPy rules:
    shapes align on the right, dimensions of size 1 repeat). A dimension that is 1
    in both keeps the output's index variable, so schedule primitives see the
    access as elementwise."""
    offset = len(out_shape) - len(shape)
    return tuple(
        IntImm(0) if d == 1 and out_shape[offset + i] != 1 else indices[offset + i]
        for i, d in enumerate(shape)
    )


def broadcast(t: TensorLike, out_shape: Sequence[int], indices: Sequence[Expr]) -> Expr:
    """``t`` read at output position ``indices`` of the broadcast shape."""
    return t[broadcast_index(out_shape, t.shape, indices)]


def clamp(e: Expr, lo: int, hi: int) -> Expr:
    """``min(max(e, lo), hi)``, keeping a load in bounds where a select discards it."""
    return minimum(maximum(e, lo), hi)
