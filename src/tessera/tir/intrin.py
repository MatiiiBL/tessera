"""Registered intrinsics: backend calls the interpreter can run and tensorize can emit.

An intrinsic has a name ``<backend>.<name>`` (``c.sgemm_8x8``), a Python
``reference`` the interpreter calls in its place, and, for tensorize, a
description ``desc`` and an ``impl``.

``desc`` is a small PrimFunc computing what the intrinsic computes: a perfect
loop nest around one block (``C[i, j] = C[i, j] + A[i, k] * B[k, j]``). Its
loop extents may be variables (a ``K`` left open); its buffers' shapes are
not used. ``Schedule.tensorize`` matches a loop nest against it and calls
``impl(match)`` with an ``IntrinMatch``: for each desc parameter the matched
buffer, the address of the first element of the matched region and the
buffer's strides, plus the matched loop extents. ``impl`` returns the
statement that replaces the nest, normally ``Evaluate(Call("c.sgemm_8x8",
...))``.

A tensorized block keeps the matched regions (``reads``: every desc
parameter in order; ``writes``: the ones the desc stores to), and the
interpreter runs it as ``reference(*views, *extents)`` with NumPy views of
those regions (writes go to the views)::

    def sgemm_ref(a, b, c, m, n, k):
        c += a @ b

A registered call that appears outside a tensorized block (written by hand)
is run as ``reference(*args)`` with the evaluated arguments: a ``Pointer``
for each ``AddressOf`` and Python numbers for the rest.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from tessera.dtype import DType
from tessera.tir.expr import AddressOf, IntImm, Var
from tessera.tir.stmt import (
    TRUE,
    Block,
    BlockRealize,
    Buffer,
    BufferRegion,
    BufferStore,
    For,
    ForKind,
    IterKind,
    IterVar,
    PrimFunc,
    Range,
    Stmt,
)

type NDArray = npt.NDArray[Any]

type Reference = Callable[..., Any]


@dataclass(frozen=True)
class IntrinMatch:
    """A loop nest matched against an intrinsic's description, one entry per desc
    parameter (inputs first, then outputs)."""

    buffers: tuple[Buffer, ...]  # the matched buffer of each desc parameter
    pointers: tuple[AddressOf, ...]  # the first element of each matched region
    strides: tuple[tuple[int, ...], ...]  # each matched buffer's strides, in elements
    regions: tuple[BufferRegion, ...]  # the matched region of each parameter
    extents: tuple[int, ...]  # the matched loops' trip counts, outermost first
    sizes: dict[str, int]  # the values of the desc's variable extents, by name


type Impl = Callable[[IntrinMatch], Stmt]


class Pointer:
    """An element of an interpreter buffer, passed to intrinsic references."""

    __slots__ = ("array", "offset")

    def __init__(self, array: NDArray, offset: int) -> None:
        self.array = array  # the whole buffer, flat
        self.offset = offset

    def view(self, shape: Sequence[int], strides: Sequence[int]) -> NDArray:
        """A writable NumPy view of ``shape`` starting here, with strides in elements."""
        if not shape:
            return self.array[self.offset : self.offset + 1].reshape(())
        last = self.offset + sum((n - 1) * s for n, s in zip(shape, strides, strict=True))
        if self.offset < 0 or last >= self.array.size or min(strides, default=0) < 0:
            raise IndexError(f"view {tuple(shape)} at {self.offset} leaves the buffer")
        item = self.array.itemsize
        return np.lib.stride_tricks.as_strided(
            self.array[self.offset :],
            shape=tuple(shape),
            strides=tuple(s * item for s in strides),
            writeable=True,
        )

    def __repr__(self) -> str:
        return f"Pointer(+{self.offset} of {self.array.size})"


@dataclass(frozen=True)
class Intrinsic:
    """A backend call with what it computes (``desc``), how to emit it (``impl``)
    and how the interpreter runs it (``reference``)."""

    name: str
    reference: Reference
    desc: PrimFunc | None = None
    impl: Impl | None = None


_REGISTRY: dict[str, Intrinsic] = {}


def register(
    name: str,
    desc: PrimFunc | None = None,
    impl: Impl | None = None,
    reference: Reference | None = None,
    *,
    override: bool = False,
) -> Intrinsic:
    """Register an intrinsic. ``reference`` is required: every intrinsic must be
    runnable by the interpreter. ``desc`` and ``impl`` are needed for tensorize."""
    if reference is None:
        raise ValueError(f"intrinsic {name} needs a Python reference")
    if name in _REGISTRY and not override:
        raise ValueError(f"intrinsic {name} is registered twice")
    if (desc is None) != (impl is None):
        raise ValueError(f"intrinsic {name}: give both desc and impl, or neither")
    intrinsic = Intrinsic(name, reference, desc, impl)
    _REGISTRY[name] = intrinsic
    return intrinsic


def get(name: str) -> Intrinsic:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown intrinsic {name!r}") from None


def registered() -> list[str]:
    return sorted(_REGISTRY)


def matmul_desc(m: int, n: int, dtype: DType = DType.float32) -> PrimFunc:
    """The description of an ``m x n`` GEMM microkernel accumulating over any ``K``:
    ``c[i, j] = c[i, j] + a[i, k] * b[k, j]`` with parameters ``(a, b, c)`` and the
    extent of ``k`` left as the variable ``K`` (``IntrinMatch.sizes["K"]``). It
    matches the update block of a decomposed matmul reduction."""
    k_extent = Var("K")
    a, b, c = Buffer("a", (m, 1), dtype), Buffer("b", (1, n), dtype), Buffer("c", (m, n), dtype)
    i, j, k = Var("i"), Var("j"), Var("k")
    vi, vj, vk = Var("vi"), Var("vj"), Var("vk")
    block = Block(
        "gemm",
        (
            IterVar(vi, Range.of(m)),
            IterVar(vj, Range.of(n)),
            IterVar(vk, Range(IntImm(0), k_extent), IterKind.REDUCE),
        ),
        (),
        (),
        BufferStore(c, c[vi, vj] + a[vi, vk] * b[vk, vj], (vi, vj)),
    )
    body: Stmt = BlockRealize((i, j, k), TRUE, block)
    body = For(k, IntImm(0), k_extent, ForKind.SERIAL, body)
    body = For(j, IntImm(0), IntImm(n), ForKind.SERIAL, body)
    body = For(i, IntImm(0), IntImm(m), ForKind.SERIAL, body)
    root = BlockRealize((), TRUE, Block("root", (), (), (), body))
    return PrimFunc(f"matmul_{m}x{n}", (a, b, c), root)
