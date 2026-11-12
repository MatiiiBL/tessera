"""The Metal backend: MSL kernels, an Objective-C runtime shim, one command buffer per run."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tessera.backend.base import Template
    from tessera.backend.metal.executor import MetalModel
    from tessera.plan import ExecutionPlan
    from tessera.target import Target
    from tessera.tir.stmt import PrimFunc


class MetalBackend:
    """Implements ``tessera.backend.base.Backend`` for the GPU."""

    name = "metal"

    def templates(self, target: Target | None = None) -> Mapping[str, Template]:
        """The GPU schedule template of every pattern that has one."""
        return {}

    def build(self, plan: ExecutionPlan, target: Target) -> MetalModel:
        """Compile every kernel of ``plan`` into one library and wire its launches."""
        from tessera.backend.metal.executor import MetalModel

        _check_target(target)
        return MetalModel(plan, target)

    def measure(
        self,
        func: PrimFunc,
        target: Target,
        *,
        repeat: int = 5,
        min_seconds: float = 0.05,
    ) -> float:
        """Median GPU seconds of one call of ``func`` on random inputs; raises
        ``MeasureError`` if it does not compile or run."""
        from tessera.backend.metal.measure import measure

        _check_target(target)
        return measure(func, repeat=repeat, min_seconds=min_seconds)

    def prebuild(self, func: PrimFunc, target: Target) -> None:
        """Generate and compile ``func`` without running it, so that the tuner's
        build processes report broken candidates; raises ``MeasureError``."""
        from tessera.backend.metal.measure import prebuild

        _check_target(target)
        prebuild(func)


def _check_target(target: Target) -> None:
    if target.kind != "metal":
        raise ValueError(f"the Metal backend cannot build for a {target.kind!r} target")


def backend() -> MetalBackend:
    """The Metal backend."""
    return MetalBackend()


def available() -> bool:
    """Whether this machine has a usable Metal GPU."""
    from tessera.backend.metal.runtime import available as runtime_available

    return runtime_available()
