"""Comparing search strategies: best time found against the number of trials.

Each strategy tunes each task from scratch several times with different
seeds. After every measurement the best time so far is divided into the
task's reference time (the exhaustive optimum for synthetic tasks, the best
time any run found for real ones), giving a curve that rises to 1.0. The
curves are averaged over seeds; the worst seed is reported too, since a
search that is good on average but sometimes stuck is not good.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tessera.bench.report import markdown_table
from tessera.tir.stmt import PrimFunc
from tessera.tune.log import TuningLog
from tessera.tune.measure import InlineRunner, Runner
from tessera.tune.search import EvolutionarySearch, RandomSearch, SearchStrategy, TaskState
from tessera.tune.space import ConfigSpace, KnobValue, config_json
from tessera.tune.synthetic import SyntheticMatmul
from tessera.tune.task import Task, TuningTask
from tessera.tune.tuner import tune

STRATEGIES: dict[str, Callable[[], SearchStrategy]] = {
    "random": RandomSearch,
    "model": EvolutionarySearch,
}
SYNTHETIC_SHAPES = ((64, 64, 64), (256, 128, 192), (32, 512, 64))


def best_so_far(state: TaskState) -> list[float]:
    """The best time after each measurement, in measurement order (inf until one runs)."""
    best, out = np.inf, []
    for key in state.order:
        seconds = state.measured[key]
        best = min(best, seconds) if seconds is not None else best
        out.append(best)
    return out


@dataclass
class Comparison:
    """Every run of every strategy on one task."""

    task: str
    reference: float  # seconds that count as 1.0
    reference_kind: str  # "optimum" or "best found"
    runs: dict[str, list[list[float]]]  # strategy -> one best-so-far curve per seed

    def relative(self, strategy: str) -> np.ndarray:
        """(seeds, trials) of reference / best-so-far."""
        curves = self.runs[strategy]
        width = min(len(c) for c in curves)
        return self.reference / np.array([c[:width] for c in curves])

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "reference_seconds": self.reference,
            "reference": self.reference_kind,
            "runs": self.runs,
        }


def compare(
    tasks: Sequence[TuningTask],
    *,
    trials: int,
    seeds: Sequence[int],
    strategies: Sequence[str] = tuple(STRATEGIES),
    runner: Runner | None = None,
    optimum: Callable[[TuningTask], float | None] = lambda task: None,
    log_dir: Path | None = None,
    on_run: Callable[[str, str, int], None] | None = None,
) -> list[Comparison]:
    """Tune every task with every strategy and seed; ``log_dir`` keeps each run's log."""
    runner = runner if runner is not None else InlineRunner()
    out: list[Comparison] = []
    for task in tasks:
        runs: dict[str, list[list[float]]] = {}
        for name in strategies:
            for seed in seeds:
                if on_run is not None:
                    on_run(task.key, name, seed)
                log = TuningLog(_log_path(log_dir, task, name, seed) if log_dir else None)
                state = tune(
                    [task], STRATEGIES[name](), trials=trials, seed=seed, runner=runner, log=log
                )[0]
                runs.setdefault(name, []).append(best_so_far(state))
        known = optimum(task)
        found = min(c[-1] for curves in runs.values() for c in curves)
        reference, kind = (known, "optimum") if known is not None else (found, "best found")
        out.append(Comparison(task.key, reference, kind, runs))
    return out


def synthetic_tasks() -> list[SyntheticMatmul]:
    return [SyntheticMatmul(m, n, k) for m, n, k in SYNTHETIC_SHAPES]


def compare_synthetic(*, trials: int = 256, seeds: Sequence[int] = range(5)) -> list[Comparison]:
    """The comparison on synthetic matmuls, whose optima are known exactly."""

    def optimum(task: TuningTask) -> float | None:
        return task.optimum()[1] if isinstance(task, SyntheticMatmul) else None

    return compare(synthetic_tasks(), trials=trials, seeds=seeds, optimum=optimum)


def checkpoints(trials: int) -> list[int]:
    """Trial counts to tabulate: powers of two from 8, and the last trial."""
    points = [t for t in (8, 16, 32, 64, 128, 256, 512, 1024) if t < trials]
    return [*points, trials]


def table(comparisons: Sequence[Comparison]) -> str:
    """Markdown: per task, mean (worst seed) of reference / best time at each checkpoint."""
    parts: list[str] = []
    for c in comparisons:
        strategies = list(c.runs)
        width = min(c.relative(s).shape[1] for s in strategies)
        rows: list[list[object]] = []
        for t in checkpoints(width):
            row: list[object] = [t]
            for s in strategies:
                rel = c.relative(s)[:, t - 1]
                row.append(f"{rel.mean():.3f} ({rel.min():.3f})")
            rows.append(row)
        seeds = len(c.runs[strategies[0]])
        parts.append(
            f"**{c.task}** (reference: {c.reference_kind}, {c.reference * 1e6:.1f} µs; "
            f"mean over {seeds} seeds, worst seed in parentheses)\n\n"
            + markdown_table(["trials", *strategies], rows)
        )
    return "\n\n".join(parts)


def trials_to_reach(c: Comparison, strategy: str, fraction: float) -> float | None:
    """Mean number of trials until reference / best >= ``fraction`` (None if a seed never does)."""
    rel = c.relative(strategy)
    hits = []
    for curve in rel:
        reached = np.flatnonzero(curve >= fraction)
        if len(reached) == 0:
            return None
        hits.append(int(reached[0]) + 1)
    return float(np.mean(hits))


def _log_path(directory: Path, task: TuningTask, strategy: str, seed: int) -> Path:
    safe = task.key.replace(":", "-")
    return directory / f"{safe}-{strategy}-seed{seed}.jsonl"


def heaviest_tasks(model: Path, target: str, count: int, *, repeat: int = 5) -> list[Task]:
    """The ``count`` tasks of ``model`` that take the most time untuned (summed over
    the groups that share each task), from a profile of the whole model."""
    from tessera.driver import lower
    from tessera.ir import random_inputs

    comp = lower(model, target=target, tuned=False)
    times = comp.build().profile(random_inputs(comp.graph), repeat=repeat)
    per_task: dict[str, float] = {}
    by_kernel = {k.name: k.task_key for k in comp.kernels}
    for t in times:
        key = by_kernel.get(t.kernel)
        if key is not None:
            per_task[key] = per_task.get(key, 0.0) + t.seconds
    ranked = sorted(per_task, key=lambda key: -per_task[key])
    return [comp.tasks[key] for key in ranked[:count]]


@dataclass(frozen=True)
class Replay:
    """A real task whose times come from measuring its whole space once.

    ``schedule`` is the real task's (the cost model sees real loop nests);
    ``measure`` looks the configuration up. Every strategy and seed then
    sees the same measurements, noise included, and the optimum is known.
    """

    task: TuningTask
    table: Mapping[str, float | None]  # config_json -> seconds (None: failed)

    @property
    def key(self) -> str:
        return self.task.key

    @property
    def target_key(self) -> str:
        return self.task.target_key

    @property
    def space(self) -> ConfigSpace:
        return self.task.space

    def schedule(self, config: Mapping[str, KnobValue]) -> PrimFunc:
        return self.task.schedule(config)

    def build(self, config: Mapping[str, KnobValue]) -> str:
        return config_json(config)

    def measure(self, built: str) -> float:
        seconds = self.table[built]
        if seconds is None:
            raise RuntimeError("this configuration failed when the space was measured")
        return seconds

    def optimum(self) -> float:
        return min(s for s in self.table.values() if s is not None)


def measure_space(task: TuningTask, runner: Runner, log: TuningLog) -> dict[str, float | None]:
    """Measure every configuration of ``task`` (resuming from ``log``)."""
    state = tune([task], RandomSearch(), trials=task.space.size, runner=runner, log=log)[0]
    return dict(state.measured)


def compare_real(
    models: Sequence[Path],
    *,
    target: str,
    tasks_per_model: int,
    trials: int,
    seeds: Sequence[int],
    log_dir: Path,
    on_run: Callable[[str, str, int], None] | None = None,
) -> list[Comparison]:
    """The comparison on the heaviest tasks of real models, replayed from one
    exhaustive measurement of each task's space on ``target``."""
    from tessera.tune.measure import LocalRunner

    tasks = [t for m in models for t in heaviest_tasks(m, target, tasks_per_model)]
    with LocalRunner() as runner:
        replays = [
            Replay(t, measure_space(t, runner, TuningLog(_log_path(log_dir, t, "all", 0))))
            for t in tasks
        ]

    def optimum(task: TuningTask) -> float | None:
        return task.optimum() if isinstance(task, Replay) else None

    return compare(replays, trials=trials, seeds=seeds, optimum=optimum, on_run=on_run)
