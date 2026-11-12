"""Search strategies and the cost model on synthetic tasks with known optima."""

from __future__ import annotations

import random

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from tessera.tune.features import extract
from tessera.tune.model import Sample, TreeCostModel, relative_throughput
from tessera.tune.search import (
    EvolutionarySearch,
    RandomSearch,
    TaskState,
    mutate,
    sample_unmeasured,
)
from tessera.tune.space import ConfigSpace, config_json
from tessera.tune.synthetic import SyntheticMatmul
from tessera.tune.tuner import tune

TASK = SyntheticMatmul(64, 64, 64)


def test_relative_throughput_is_per_task() -> None:
    v = np.zeros(1)
    samples = [
        Sample("a", v, 2.0),
        Sample("a", v, 1.0),
        Sample("b", v, 10.0),
        Sample("b", v, None),
        Sample("b", v, 40.0),
    ]
    assert relative_throughput(samples).tolist() == [0.5, 1.0, 1.0, 0.0, 0.25]


def test_cost_model_ranks_unseen_candidates() -> None:
    rng = random.Random(0)
    train = [TASK.space.sample(rng) for _ in range(160)]
    model = TreeCostModel()
    model.fit([Sample(TASK.key, extract(TASK.schedule(p)), TASK.cost(p)) for p in train])
    assert model.trained
    pool = [TASK.space.sample(rng) for _ in range(1000)]
    scores = model.predict(np.stack([extract(TASK.schedule(p)) for p in pool]))
    costs = np.array([TASK.cost(p) for p in pool])
    top = np.argsort(-scores)[:20]
    # The model's top 20 are mostly among the fastest tenth of the pool.
    assert np.mean(costs[top] <= np.quantile(costs, 0.1)) >= 0.5
    ranks = np.argsort(np.argsort(costs))
    spearman = np.corrcoef(np.argsort(np.argsort(-scores)), ranks)[0, 1]
    assert spearman > 0.5


def test_cost_model_needs_data() -> None:
    model = TreeCostModel(min_samples=8)
    model.fit([Sample("t", np.zeros(3), 1.0)] * 4)
    assert not model.trained
    assert model.predict(np.zeros((2, 3))).tolist() == [0.0, 0.0]


@settings(max_examples=200, deadline=None)
@given(st.integers(0, TASK.space.size - 1), st.integers(0, 2**32))
def test_mutation_changes_a_knob_and_stays_in_the_space(index: int, seed: int) -> None:
    point = TASK.space.get(index)
    child = mutate(point, TASK.space.knobs, random.Random(seed))
    assert child != point
    assert all(child[k] in TASK.space.knobs[k] for k in TASK.space.knobs)


def test_sampling_without_replacement_exhausts_small_spaces() -> None:
    space = ConfigSpace({"a": [1, 2, 3], "b": [True, False]}, {"a": 1, "b": True})

    class Tiny:
        key = "tiny"
        target_key = "none"

        def __init__(self) -> None:
            self.space = space

    state = TaskState(Tiny())  # type: ignore[arg-type]
    state.record({"a": 1, "b": True}, 1.0)
    rest = sample_unmeasured(state, 10, random.Random(0))
    assert len(rest) == 5
    assert len({config_json(p) for p in rest}) == 5
    assert not any(state.is_measured(p) for p in rest)


def test_proposals_are_distinct_and_unmeasured() -> None:
    rng = random.Random(1)
    for strategy in (RandomSearch(), EvolutionarySearch()):
        state = TaskState(TASK)
        for p in RandomSearch().propose(state, 32, rng):
            state.record(p, TASK.cost(p))
        strategy.update([state])
        batch = strategy.propose(state, 16, rng)
        assert len(batch) == 16
        assert len({config_json(p) for p in batch}) == 16
        assert not any(state.is_measured(p) for p in batch)


def best_curve(strategy_name: str, seed: int, trials: int) -> list[float]:
    strategy = EvolutionarySearch() if strategy_name == "model" else RandomSearch()
    state = tune([TASK], strategy, trials=trials, seed=seed)[0]
    seconds = [state.measured[k] for k in state.order]
    return np.minimum.accumulate([s if s is not None else np.inf for s in seconds]).tolist()


def test_search_is_deterministic_for_a_seed() -> None:
    assert best_curve("model", 3, 64) == best_curve("model", 3, 64)


def test_model_guided_search_beats_random_search() -> None:
    _, optimum = TASK.optimum()
    trials = 128
    model = np.mean([optimum / best_curve("model", s, trials)[-1] for s in range(3)])
    rand = np.mean([optimum / best_curve("random", s, trials)[-1] for s in range(3)])
    assert model > 0.9
    assert model > rand + 0.05
