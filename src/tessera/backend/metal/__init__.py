"""The Metal backend: MSL kernels, an Objective-C runtime shim, one command buffer per run."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tessera.backend.base import Backend


def backend() -> Backend:
    raise NotImplementedError("the Metal backend is not built yet")
