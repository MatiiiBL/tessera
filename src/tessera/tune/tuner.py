"""The tuning loop: rounds of proposing, measuring, logging and learning.

``tune`` works on any number of tasks with one shared budget of
measurements. Each round picks one task, measures a batch of its
configurations (the template's default first, so every task has a
baseline), appends every result to the log, and retrains the strategy's
cost model on all measurements of all tasks.

The round goes to the task with the largest expected saving of model time:
``weight * best_seconds * rate``, where ``weight`` is how many groups of the
model share the task and ``rate`` is the task's relative improvement per
round over its last two rounds (optimistically 0.5 until it has had two,
and at least 0.005 so a stalled task can still be revisited). Tasks with
nothing measured go first; tasks whose space is exhausted are skipped.

The budget counts the log: a run with ``trials=500`` that finds 300 of its
tasks' measurements already logged measures 200 more, so an interrupted
run resumes where it stopped by running the same command again.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from tessera.tune.log import Record, TuningLog, now
from tessera.tune.measure import InlineRunner, MeasureResult, Runner
from tessera.tune.search import SearchStrategy, TaskState
from tessera.tune.space import config_json
from tessera.tune.task import Point, TuningTask

MIN_RATE = 0.005
OPTIMISTIC_RATE = 0.5


@dataclass
class Round:
    """What one round measured, for progress reports."""

    index: int
    task: TuningTask
    results: list[MeasureResult]
    state: TaskState
    measured_total: int
    trials: int


@dataclass
class TaskProgress:
    state: TaskState
    weight: float
    history: list[float] = field(default_factory=list)  # best seconds after each round
    stalled: bool = False  # the strategy found nothing left to propose

    def priority(self) -> float:
        if not self.state.measured:
            return float("inf")
        if self.state.exhausted or self.stalled:
            return float("-inf")
        best = self.state.best()
        if best is None:
            return 0.0  # nothing has run yet: only when no other task can use the round
        if len(self.history) < 2:
            rate = OPTIMISTIC_RATE
        else:
            before = self.history[-3] if len(self.history) >= 3 else self.history[0]
            rate = (before - self.history[-1]) / before / 2.0
        return self.weight * best[1] * max(rate, MIN_RATE)


def tune(
    tasks: Sequence[TuningTask],
    strategy: SearchStrategy,
    *,
    trials: int,
    runner: Runner | None = None,
    log: TuningLog | None = None,
    batch_size: int = 16,
    seed: int = 0,
    weights: Sequence[float] | None = None,
    time_budget: float | None = None,
    on_round: Callable[[Round], None] | None = None,
) -> list[TaskState]:
    """Tune ``tasks`` until ``trials`` measurements (log included) or ``time_budget``
    seconds; returns what is known about each task."""
    runner = runner if runner is not None else InlineRunner()
    log = log if log is not None else TuningLog()
    rng = random.Random(seed)
    weights = weights if weights is not None else [float(getattr(t, "weight", 1)) for t in tasks]
    progress = [TaskProgress(_resume(task, log), w) for task, w in zip(tasks, weights, strict=True)]
    strategy.update([p.state for p in progress])
    started = time.monotonic()
    round_index = 0
    while True:
        total = sum(len(p.state.measured) for p in progress)
        if total >= trials:
            break
        if time_budget is not None and time.monotonic() - started >= time_budget:
            break
        p = max(progress, key=lambda p: p.priority())
        if p.priority() == float("-inf"):
            break
        configs = _batch(p.state, strategy, min(batch_size, trials - total), rng)
        if not configs:
            p.stalled = True
            continue
        results = runner.run(p.state.task, configs)
        for r in results:
            p.state.record(r.config, r.seconds)
            log.append(
                Record(
                    p.state.task.target_key, p.state.task.key, r.config, r.seconds, r.error, now()
                )
            )
        best = p.state.best()
        p.history.append(best[1] if best is not None else float("inf"))
        strategy.update([q.state for q in progress])
        if on_round is not None:
            on_round(
                Round(round_index, p.state.task, results, p.state, total + len(results), trials)
            )
        round_index += 1
    return [p.state for p in progress]


def _resume(task: TuningTask, log: TuningLog) -> TaskState:
    state = TaskState(task)
    for record in log.measured(task.target_key, task.key).values():
        state.record(record.config, record.seconds if record.ok else None)
    return state


def _batch(state: TaskState, strategy: SearchStrategy, n: int, rng: random.Random) -> list[Point]:
    configs: list[Point] = []
    default = state.task.space.default()
    if not state.is_measured(default):
        configs.append(default)
    seen = {config_json(c) for c in configs}
    for c in strategy.propose(state, n - len(configs), rng):
        key = config_json(c)
        if key not in seen and not state.is_measured(c):
            seen.add(key)
            configs.append(c)
    return configs[:n]
