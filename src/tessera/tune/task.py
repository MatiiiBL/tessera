"""Tuning tasks: what the tuner searches over.

A task is one lowered group's structure plus the template that schedules it
and the target it runs on. Two groups with the same structure (the same
shapes and computation, whatever their names) are the same task, so a model
with twenty identical convolutions tunes that convolution once.

The tuner only needs the ``TuningTask`` protocol, which the synthetic tasks
in ``tessera.tune.synthetic`` also implement, so the search strategies and
the cost model are tested without a backend.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from tessera.plan import LoweredGroup
from tessera.target import Target
from tessera.tir.schedule import Schedule
from tessera.tir.stmt import PrimFunc
from tessera.tune.space import ApplyConfig, Config, ConfigSpace, KnobValue, space_of

if TYPE_CHECKING:
    from tessera.backend.base import Template

type Point = dict[str, KnobValue]


class TuningTask(Protocol):
    """A search problem: a space of configurations and a way to time each one.

    ``build`` runs in a build process and ``measure`` in the measurement
    process (see ``tessera.tune.measure``), so the task must be picklable and
    so must ``build``'s result.
    """

    @property
    def key(self) -> str:
        """Identity in tuning logs: ``<pattern>:<structural hash>``."""
        ...

    @property
    def target_key(self) -> str:
        """The target identity in tuning logs (``Target.key``)."""
        ...

    @property
    def space(self) -> ConfigSpace: ...

    def schedule(self, config: Mapping[str, KnobValue]) -> PrimFunc:
        """The function scheduled with ``config`` (features come from it)."""
        ...

    def build(self, config: Mapping[str, KnobValue]) -> Any:
        """Prepare ``config`` for measurement; raises if it cannot be built."""
        ...

    def measure(self, built: Any) -> float:
        """Seconds per call of what ``build`` returned; raises if it fails."""
        ...


def structural_hash(func: PrimFunc) -> str:
    """The loop IR's structural hash of ``func`` (``tessera.tir.structural_hash``)."""
    import tessera.tir

    # Resolved at call time: the hash belongs to the tir package, which this
    # module must not import eagerly (the tuner works without it).
    fn: Callable[[PrimFunc], str] = tessera.tir.__dict__["structural_hash"]
    return fn(func)


def task_key(lowered: LoweredGroup) -> str:
    """``<pattern>:<structural hash>`` of a lowered group (names do not matter)."""
    return f"{lowered.pattern}:{structural_hash(lowered.func)[:16]}"


@dataclass
class Task:
    """A lowered group, the template for its pattern, and a target.

    ``weight`` counts the groups of the model that share this task, so the
    tuner can spend its budget where the model spends its time. ``measure``
    times a candidate with the target's backend (``Backend.measure``).
    """

    lowered: LoweredGroup
    template: Template
    target: Target
    key: str
    space: ConfigSpace
    weight: int = 1
    names: list[str] = field(default_factory=list)  # the groups that share the task
    repeat: int = 5
    min_seconds: float = 0.05

    @classmethod
    def create(
        cls, lowered: LoweredGroup, template: Template, target: Target, *, key: str | None = None
    ) -> Task:
        """The task of ``lowered``; its space is recorded by running ``template`` once."""

        def define(cfg: Config) -> None:
            template(Schedule(lowered.func), lowered, cfg)

        return cls(
            lowered,
            template,
            target,
            key if key is not None else task_key(lowered),
            space_of(define),
            names=[lowered.group.name],
        )

    @property
    def target_key(self) -> str:
        return self.target.key

    def schedule(self, config: Mapping[str, KnobValue]) -> PrimFunc:
        sch = Schedule(self.lowered.func)
        self.template(sch, self.lowered, ApplyConfig(config))
        return sch.func

    def build(self, config: Mapping[str, KnobValue]) -> PrimFunc:
        """Schedule ``config``; backends with a ``prebuild`` method also compile it
        here, so compilation happens in the parallel build processes."""
        func = self.schedule(config)
        prebuild = getattr(self._backend(), "prebuild", None)
        if callable(prebuild):
            prebuild(func, self.target)
        return func

    def measure(self, built: PrimFunc) -> float:
        backend = self._backend()
        return float(
            backend.measure(built, self.target, repeat=self.repeat, min_seconds=self.min_seconds)
        )

    def _backend(self) -> Any:
        from tessera.backend.base import get_backend

        return get_backend(self.target.kind)


def extract_tasks(
    lowered: Sequence[LoweredGroup],
    templates: Mapping[str, Template],
    target: Target,
    *,
    key: Callable[[LoweredGroup], str] = task_key,
) -> list[Task]:
    """One task per distinct structure, in first-appearance order, with weights."""
    tasks: dict[str, Task] = {}
    for group in lowered:
        k = key(group)
        if k in tasks:
            tasks[k].weight += 1
            tasks[k].names.append(group.group.name)
        else:
            tasks[k] = Task.create(group, templates[group.pattern], target, key=k)
    return list(tasks.values())
