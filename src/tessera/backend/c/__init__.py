"""The C backend: C source, clang, a thread pool, NEON and SME microkernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tessera.backend.base import Backend


def backend() -> Backend:
    raise NotImplementedError("the C backend is not built yet")
