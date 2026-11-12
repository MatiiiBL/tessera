"""What every backend provides: schedule templates, a builder and a timer."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from tessera.ir.graph import NDArray, TensorType
from tessera.plan import ExecutionPlan, LoweredGroup
from tessera.target import Target
from tessera.tir.schedule import Schedule
from tessera.tir.stmt import PrimFunc
from tessera.tune.space import Config

# A template schedules one lowered group, asking ``Config`` for every decision.
type Template = Callable[[Schedule, LoweredGroup, Config], None]


@dataclass(frozen=True)
class KernelTime:
    kernel: str
    seconds: float  # median over the profiled runs


class CompiledModel(Protocol):
    """A model ready to run on one target."""

    @property
    def target(self) -> Target: ...

    @property
    def inputs(self) -> list[tuple[str, TensorType]]: ...

    @property
    def outputs(self) -> list[tuple[str, TensorType]]: ...

    def run(self, inputs: Mapping[str, NDArray]) -> dict[str, NDArray]:
        """Run once; inputs and outputs are NumPy arrays (outputs are new arrays)."""
        ...

    def benchmark(
        self, inputs: Mapping[str, NDArray], *, warmup: int = 3, repeat: int = 20
    ) -> list[float]:
        """Wall-clock seconds of each timed run, inputs already in place."""
        ...

    def profile(self, inputs: Mapping[str, NDArray], *, repeat: int = 10) -> list[KernelTime]:
        """Per-kernel times, in launch order."""
        ...

    def sources(self) -> dict[str, str]:
        """Generated code by file name (``model.c``, ``kernels.metal``)."""
        ...


class Backend(Protocol):
    name: str

    def templates(self, target: Target | None = None) -> Mapping[str, Template]:
        """One template per ``tessera.plan.PATTERNS`` key, specialized for
        ``target`` (its ISA and thread count); None means ``Target.host``
        of this backend's kind. The driver and the tuner pass the target
        they compile or tune for."""
        ...

    def build(self, plan: ExecutionPlan, target: Target) -> CompiledModel:
        """Compile ``plan.funcs`` and wire them to ``plan``'s launches and memory."""
        ...

    def measure(
        self,
        func: PrimFunc,
        target: Target,
        *,
        repeat: int = 5,
        min_seconds: float = 0.05,
    ) -> float:
        """Median seconds of one call of ``func`` on random inputs (for tuning).
        Raises ``MeasureError`` if it does not build or run."""
        ...


@runtime_checkable
class Prebuilder(Protocol):
    """An optional capability of a backend, outside ``Backend`` so that backends
    without it still satisfy that protocol. The tuner calls ``prebuild`` in its
    parallel build processes when the backend has it, so that compilation runs
    in parallel and ``measure`` only has to run the cached result."""

    def prebuild(self, func: PrimFunc, target: Target) -> None:
        """Compile ``func`` for ``measure`` into the build cache without running it;
        safe to call from another process. Raises ``MeasureError`` if it does not
        build."""
        ...


class MeasureError(Exception):
    """A candidate failed to build, crashed or timed out while being measured."""


def get_backend(kind: str) -> Backend:
    if kind == "c":
        from tessera.backend.c import backend as c_backend

        return c_backend()
    if kind == "metal":
        from tessera.backend.metal import backend as metal_backend

        return metal_backend()
    raise ValueError(f"unknown backend {kind!r}")
