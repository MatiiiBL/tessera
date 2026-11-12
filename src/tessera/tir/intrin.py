"""Registered intrinsics: backend calls the interpreter can run and tensorize can emit.

An intrinsic has a name ``<backend>.<name>`` (``c.sgemm_8x8``), a Python
``reference`` the interpreter calls in its place, and, for tensorize, a
description ``desc`` (a small PrimFunc of what it computes) and an ``impl``
that builds the call replacing a matched loop nest.

The reference receives the evaluated call arguments: a ``Pointer`` for each
``AddressOf`` argument and Python numbers for the others. Pointers give NumPy
views into the interpreter's buffers, so a reference writes its results in
place::

    def sgemm_ref(c, a, b, lda, ldb, ldc):
        A = a.view((8, 8), (lda, 1))
        B = b.view((8, 8), (ldb, 1))
        C = c.view((8, 8), (ldc, 1))
        C += A @ B
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from tessera.tir.stmt import BufferRegion, PrimFunc, Stmt

type NDArray = npt.NDArray[Any]

type Reference = Callable[..., Any]
type Impl = Callable[[Sequence[BufferRegion]], Stmt]


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
