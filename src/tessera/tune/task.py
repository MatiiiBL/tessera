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

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from tessera.plan import LoweredGroup
from tessera.target import Target
from tessera.tir import structural_hash
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


def constant_inputs(lowered: LoweredGroup) -> tuple[int, ...]:
    """Indices of the group's inputs that are constants (weights)."""
    return tuple(i for i, v in enumerate(lowered.group.inputs) if v.is_constant)


def merge_task(tasks: dict[str, Task], task: Task) -> None:
    """Add ``task`` to ``tasks`` by key, or count it in the task with its key."""
    known = tasks.get(task.key)
    if known is None:
        tasks[task.key] = task
    else:
        known.weight += task.weight
        known.names += task.names


def task_key(lowered: LoweredGroup) -> str:
    """``<pattern>:<hash>`` of a lowered group: the structural hash of its function
    (names do not matter) and, if it has any, which of its inputs are constants.

    Backends treat constants differently from other inputs (a model prepares
    its weights once, when it is loaded), so a matmul by a weight and a
    matmul of two activations of the same shapes are different tasks.
    """
    digest = structural_hash(lowered.func)
    constants = constant_inputs(lowered)
    if constants:
        text = f"{digest} constants {','.join(map(str, constants))}"
        digest = hashlib.sha256(text.encode()).hexdigest()
    return f"{lowered.pattern}:{digest[:16]}"


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
    constants: tuple[int, ...] = ()  # indices of the group's inputs that are constants

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
            constants=constant_inputs(lowered),
        )

    @property
    def target_key(self) -> str:
        return self.target.key

    def schedule(self, config: Mapping[str, KnobValue]) -> PrimFunc:
        sch = Schedule(self.lowered.func)
        self.template(sch, self.lowered, ApplyConfig(config))
        return sch.func

    def build(self, config: Mapping[str, KnobValue]) -> PrimFunc:
        """Schedule ``config``; a backend that is a ``Prebuilder`` also compiles it
        here, so compilation happens in the parallel build processes."""
        from tessera.backend.base import Prebuilder

        func = self.schedule(config)
        backend = self._backend()
        if isinstance(backend, Prebuilder):
            backend.prebuild(func, self.target, constants=frozenset(self.constants))
        return func

    def measure(self, built: PrimFunc) -> float:
        backend = self._backend()
        return float(
            backend.measure(
                built,
                self.target,
                repeat=self.repeat,
                min_seconds=self.min_seconds,
                constants=frozenset(self.constants),
            )
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
        merge_task(tasks, Task.create(group, templates[group.pattern], target, key=key(group)))
    return list(tasks.values())
