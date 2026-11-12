"""Search strategies: which configurations to measure next.

``RandomSearch`` draws unmeasured configurations uniformly.

``EvolutionarySearch`` is model-guided, in the spirit of Ansor: a population
seeded with the best measured configurations and random ones evolves for a
few generations by mutation (one knob moves to a nearby or a random option)
and crossover (each knob from one of two parents), ranked by the cost model,
which is cheap compared with a measurement. The best unmeasured candidates
seen during the evolution are measured, together with a few random ones so
the model keeps seeing parts of the space it ranks badly; then the model is
retrained on everything measured so far. Until the model has enough data it
proposes random configurations.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from tessera.tune.features import FeatureVector, extract
from tessera.tune.model import CostModel, Sample, TreeCostModel
from tessera.tune.space import KnobValue, config_json
from tessera.tune.task import Point, TuningTask


@dataclass
class TaskState:
    """What the search knows about one task: measurements and cached features."""

    task: TuningTask
    measured: dict[str, float | None] = field(default_factory=dict)  # config json -> seconds
    points: dict[str, Point] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)  # measurement order
    featurize: Callable[[Sequence[Mapping[str, KnobValue]]], list[FeatureVector | None]] | None = (
        None
    )
    _features: dict[str, FeatureVector | None] = field(default_factory=dict, repr=False)

    def record(self, config: Mapping[str, KnobValue], seconds: float | None) -> None:
        key = config_json(config)
        if key not in self.measured:
            self.order.append(key)
        self.measured[key] = seconds
        self.points[key] = dict(config)

    def is_measured(self, config: Mapping[str, KnobValue]) -> bool:
        return config_json(config) in self.measured

    @property
    def exhausted(self) -> bool:
        return len(self.measured) >= self.task.space.size

    def best(self) -> tuple[Point, float] | None:
        """The fastest measured configuration and its time."""
        timed = [(s, k) for k, s in self.measured.items() if s is not None]
        if not timed:
            return None
        seconds, key = min(timed)
        return self.points[key], seconds

    def ranked(self) -> list[tuple[Point, float]]:
        """Measured configurations that ran, fastest first."""
        timed = sorted((s, k) for k, s in self.measured.items() if s is not None)
        return [(self.points[k], s) for s, k in timed]

    def features(self, config: Mapping[str, KnobValue]) -> FeatureVector | None:
        """Features of ``config``'s scheduled function; None if the template rejects it."""
        return self.features_many([config])[0]

    def features_many(
        self, configs: Sequence[Mapping[str, KnobValue]]
    ) -> list[FeatureVector | None]:
        """Features of several configurations; the missing ones are computed by
        ``featurize`` (in parallel processes) when it is set, else here."""
        missing = {config_json(c): c for c in configs if config_json(c) not in self._features}
        if missing:
            todo = list(missing.values())
            if self.featurize is not None:
                vectors = self.featurize(todo)
            else:
                vectors = [features_of(self.task, c) for c in todo]
            for key, vector in zip(missing, vectors, strict=True):
                self._features[key] = vector
        return [self._features[config_json(c)] for c in configs]

    def samples(self) -> list[Sample]:
        """Training samples for the cost model, in measurement order."""
        out: list[Sample] = []
        for key in self.order:
            vector = self.features(self.points[key])
            if vector is not None:
                out.append(Sample(self.task.key, vector, self.measured[key]))
        return out


def features_of(task: TuningTask, config: Mapping[str, KnobValue]) -> FeatureVector | None:
    """Features of ``config``'s scheduled function; None if the template rejects it."""
    try:
        return extract(task.schedule(config))
    except Exception:  # a template may reject a point of its own space
        return None


class SearchStrategy(Protocol):
    name: str

    def propose(self, state: TaskState, n: int, rng: random.Random) -> list[Point]:
        """Up to ``n`` distinct unmeasured configurations."""
        ...

    def update(self, states: Sequence[TaskState]) -> None:
        """Learn from every measurement so far (called after each round)."""
        ...


def sample_unmeasured(
    state: TaskState, n: int, rng: random.Random, exclude: set[str] | None = None
) -> list[Point]:
    """Up to ``n`` distinct random configurations not yet measured."""
    space = state.task.space
    taken = set(state.measured) | (exclude or set())
    out: list[Point] = []
    attempts = 0
    while len(out) < n and attempts < 50 * n:
        attempts += 1
        point = space.sample(rng)
        key = config_json(point)
        if key not in taken:
            taken.add(key)
            out.append(point)
    if len(out) < n and space.size <= 200_000:
        rest = [i for i in range(space.size) if config_json(space.get(i)) not in taken]
        rng.shuffle(rest)
        out.extend(space.get(i) for i in rest[: n - len(out)])
    return out


class RandomSearch:
    """Uniform sampling without replacement."""

    name = "random"

    def propose(self, state: TaskState, n: int, rng: random.Random) -> list[Point]:
        return sample_unmeasured(state, n, rng)

    def update(self, states: Sequence[TaskState]) -> None:
        return None


class EvolutionarySearch:
    """Model-guided evolutionary search (see the module docstring)."""

    name = "model"

    def __init__(
        self,
        model: CostModel | None = None,
        *,
        population: int = 128,
        generations: int = 4,
        crossover: float = 0.5,
        explore: float = 0.2,
        elite: float = 0.2,
        seeds_from_measured: int = 16,
        refit_growth: float = 0.1,
    ) -> None:
        self.model: CostModel = model if model is not None else TreeCostModel()
        self.population = population
        self.generations = generations
        self.crossover = crossover
        self.explore = explore
        self.elite = elite
        self.seeds_from_measured = seeds_from_measured
        self.refit_growth = refit_growth
        self._fitted_on = 0

    def update(self, states: Sequence[TaskState]) -> None:
        """Refit once the measurements have grown by ``refit_growth`` (at least 16)
        since the last fit: with thousands of samples, one round adds little,
        and a fit costs about a second."""
        count = sum(len(state.measured) for state in states)
        if self.model.trained and count - self._fitted_on < max(
            16, self.refit_growth * self._fitted_on
        ):
            return
        self.model.fit([s for state in states for s in state.samples()])
        self._fitted_on = count

    def propose(self, state: TaskState, n: int, rng: random.Random) -> list[Point]:
        if not self.model.trained:
            return sample_unmeasured(state, n, rng)
        scores: dict[str, float] = {}
        points: dict[str, Point] = {}

        def score(population: list[Point]) -> list[float]:
            fresh = list(
                {config_json(p): p for p in population if config_json(p) not in scores}.values()
            )
            vectors = state.features_many(fresh)
            valid = [(p, v) for p, v in zip(fresh, vectors, strict=True) if v is not None]
            if valid:
                predicted = self.model.predict(np.stack([v for _, v in valid]))
                for (p, _), s in zip(valid, predicted, strict=True):
                    scores[config_json(p)] = float(s)
                    points[config_json(p)] = p
            for p, v in zip(fresh, vectors, strict=True):
                if v is None:
                    scores[config_json(p)] = -np.inf
            return [scores[config_json(p)] for p in population]

        seeds = [p for p, _ in state.ranked()[: self.seeds_from_measured]]
        population = seeds + [
            state.task.space.sample(rng) for _ in range(self.population - len(seeds))
        ]
        for _ in range(self.generations):
            population = self._evolve(state, population, score(population), rng)
        score(population)

        ranked = sorted(
            (s, k) for k, s in scores.items() if np.isfinite(s) and k not in state.measured
        )
        take = n - max(1, round(self.explore * n)) if n > 1 else n
        chosen = [points[k] for _, k in reversed(ranked[-take:])] if take > 0 else []
        chosen_keys = {config_json(p) for p in chosen}
        chosen += sample_unmeasured(state, n - len(chosen), rng, exclude=chosen_keys)
        return chosen

    def _evolve(
        self, state: TaskState, population: list[Point], scores: list[float], rng: random.Random
    ) -> list[Point]:
        order = sorted(range(len(population)), key=lambda i: -scores[i])
        alive = [i for i in order if np.isfinite(scores[i])] or order
        n_elite = max(1, int(self.elite * len(population)))
        children = [population[i] for i in alive[:n_elite]]
        seen = {config_json(p) for p in children}
        knobs = state.task.space.knobs
        attempts = 0
        while len(children) < self.population and attempts < 10 * self.population:
            attempts += 1
            a = population[_tournament(alive, scores, rng)]
            if rng.random() < self.crossover:
                b = population[_tournament(alive, scores, rng)]
                child = {name: (a if rng.random() < 0.5 else b)[name] for name in knobs}
            else:
                child = dict(a)
            child = mutate(child, knobs, rng)
            key = config_json(child)
            if key not in seen:
                seen.add(key)
                children.append(child)
        return children


def mutate(
    point: Mapping[str, KnobValue], knobs: Mapping[str, Sequence[KnobValue]], rng: random.Random
) -> Point:
    """Change at least one knob: to a neighbouring option (options are ordered, so
    neighbouring tile sizes are close) or, half of the time, to any option."""
    names = [name for name, options in knobs.items() if len(options) > 1]
    out = dict(point)
    if not names:
        return out
    chosen = [name for name in names if rng.random() < 1.0 / len(names)] or [rng.choice(names)]
    for name in chosen:
        options = list(knobs[name])
        i = options.index(out[name])
        if rng.random() < 0.5:
            j = i + rng.choice((-2, -1, 1, 2))
            j = min(max(j, 0), len(options) - 1)
            if j == i:
                j = (i + 1) % len(options)
        else:
            j = rng.randrange(len(options) - 1)
            j = j + 1 if j >= i else j
        out[name] = options[j]
    return out


def _tournament(alive: list[int], scores: list[float], rng: random.Random) -> int:
    a, b = rng.choice(alive), rng.choice(alive)
    return a if scores[a] >= scores[b] else b
