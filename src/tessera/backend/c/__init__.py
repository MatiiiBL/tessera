"""The C backend: C source, clang, a thread pool, NEON and SME microkernels.

``backend()`` returns the ``tessera.backend.base.Backend`` of this package.
docs/cpu.md describes the generated code, the runtime and the measurements.
"""

from __future__ import annotations

from collections.abc import Mapping

from tessera.backend.base import Backend, Template
from tessera.backend.c.executor import CModel, build_model, run_kernel
from tessera.backend.c.measure import DEFAULT_TIMEOUT, build_harness
from tessera.backend.c.measure import measure as measure_kernel
from tessera.backend.c.templates import make_templates
from tessera.plan import ExecutionPlan
from tessera.target import Target
from tessera.tir.stmt import PrimFunc


class CBackend:
    """Builds ``ExecutionPlan``s into shared libraries and times single kernels."""

    name = "c"

    def templates(self, target: Target | None = None) -> Mapping[str, Template]:
        """One schedule template per pattern of ``tessera.plan.PATTERNS``,
        specialized for ``target`` (the host when None)."""
        return make_templates(target or Target.host())

    def build(self, plan: ExecutionPlan, target: Target) -> CModel:
        """Compile every kernel of ``plan`` into one library and wire it to the plan."""
        _check_target(target)
        return build_model(plan, target)

    def measure(
        self,
        func: PrimFunc,
        target: Target,
        *,
        repeat: int = 5,
        min_seconds: float = 0.05,
        constants: frozenset[int] = frozenset(),
        timeout: float = DEFAULT_TIMEOUT,
    ) -> float:
        """Median seconds of one call of ``func`` on random inputs, timed in a
        separate process; ``MeasureError`` if it does not build, crashes or
        exceeds ``timeout``. The parameters at the indices ``constants`` hold
        model constants: they are prepared (packed) before timing, as in a
        built model, and read cold, as a model run reads its weights."""
        _check_target(target)
        return measure_kernel(
            func,
            target,
            repeat=repeat,
            min_seconds=min_seconds,
            timeout=timeout,
            constants=constants,
        )

    def prebuild(
        self, func: PrimFunc, target: Target, *, constants: frozenset[int] = frozenset()
    ) -> None:
        """Compile ``func``'s timing executable for ``measure`` with the same
        ``constants`` into the build cache without running it. Safe to call
        from several processes at once; ``MeasureError`` if it does not build."""
        _check_target(target)
        build_harness(func, target, constants)


def _check_target(target: Target) -> None:
    if target.kind != "c":
        raise ValueError(f"the C backend cannot build for a {target.kind!r} target")


def backend() -> Backend:
    """The C backend."""
    return CBackend()


__all__ = ["CBackend", "CModel", "backend", "run_kernel"]
