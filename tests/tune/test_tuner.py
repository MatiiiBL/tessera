"""The tuning loop: budgets, logging, resuming and the task scheduler."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from tessera.tune.log import BestConfigs, TuningLog
from tessera.tune.measure import InlineRunner
from tessera.tune.search import EvolutionarySearch, RandomSearch
from tessera.tune.space import KnobValue, config_json
from tessera.tune.synthetic import SyntheticMatmul
from tessera.tune.tuner import Round, tune


@dataclass(frozen=True)
class Picky(SyntheticMatmul):
    """Fails to build every configuration without vectorization."""

    def build(self, config: Mapping[str, KnobValue]) -> dict[str, KnobValue]:
        if not config["vectorize"]:
            raise RuntimeError("scalar loop rejected")
        return dict(config)


def test_budget_default_first_and_every_result_logged() -> None:
    task = SyntheticMatmul(32, 32, 32)
    log = TuningLog()
    rounds: list[Round] = []
    state = tune([task], RandomSearch(), trials=40, log=log, on_round=rounds.append)[0]
    assert len(state.measured) == 40
    assert state.order[0] == config_json(task.space.default())
    assert [r.measured_total for r in rounds] == [16, 32, 40]
    records = log.records()
    assert len(records) == 40
    assert len({config_json(r.config) for r in records}) == 40
    assert all(r.target == task.target_key and r.task == task.key for r in records)


def test_resume_skips_measured_configurations(tmp_path: Path) -> None:
    task = SyntheticMatmul(32, 32, 32)
    path = tmp_path / "tune.jsonl"
    first = tune([task], EvolutionarySearch(), trials=48, log=TuningLog(path), seed=1)[0]
    second = tune([task], EvolutionarySearch(), trials=96, log=TuningLog(path), seed=2)[0]
    records = TuningLog(path).records()
    assert len(records) == 96
    assert len({config_json(r.config) for r in records}) == 96
    assert set(first.measured) <= set(second.measured)
    # Nothing left to do: the same command again measures nothing.
    third = tune([task], EvolutionarySearch(), trials=96, log=TuningLog(path))[0]
    assert len(TuningLog(path).records()) == 96
    best = BestConfigs.load(path).lookup(task.target_key, task.key)
    assert best is not None
    assert (dict(best.config), best.seconds) == third.best()


def test_failures_are_recorded_not_fatal() -> None:
    task = Picky(32, 32, 32)
    log = TuningLog()
    state = tune([task], EvolutionarySearch(), trials=64, log=log)[0]
    failed = [r for r in log.records() if not r.ok]
    assert failed
    assert all("scalar loop rejected" in (r.error or "") for r in failed)
    best = state.best()
    assert best is not None
    assert best[0]["vectorize"] is True
    assert len(state.measured) == 64


def test_scheduler_spends_the_budget_where_the_time_is() -> None:
    big = SyntheticMatmul(128, 128, 128)
    small = SyntheticMatmul(16, 16, 16)
    rounds: list[Round] = []
    states = tune([small, big], RandomSearch(), trials=160, batch_size=16, on_round=rounds.append)
    counts = Counter(r.task.key for r in rounds)
    assert [r.task.key for r in rounds[:2]] == [small.key, big.key]  # untuned tasks first
    assert counts[big.key] > counts[small.key]
    assert sum(len(s.measured) for s in states) == 160
    # A weight makes a task matter more: 50 copies of the small one win the budget.
    weighted = tune(
        [small, big], RandomSearch(), trials=160, weights=[200.0, 1.0], on_round=rounds.append
    )
    assert len(weighted[0].measured) > len(weighted[1].measured)


def test_exhausted_spaces_stop_the_run() -> None:
    task = SyntheticMatmul(2, 2, 2)
    state = tune([task], RandomSearch(), trials=10_000, batch_size=64)[0]
    assert len(state.measured) == task.space.size
    assert state.exhausted


def test_time_budget() -> None:
    task = SyntheticMatmul(64, 64, 64)
    state = tune([task], RandomSearch(), trials=10_000, time_budget=0.0)[0]
    assert len(state.measured) == 0


@dataclass(frozen=True)
class Broken(SyntheticMatmul):
    """Nothing builds."""

    def build(self, config: Mapping[str, KnobValue]) -> dict[str, KnobValue]:
        raise RuntimeError("template bug")


def test_a_task_where_nothing_runs_does_not_take_the_budget() -> None:
    broken, fine = Broken(32, 32, 32), SyntheticMatmul(32, 32, 32, threads=2)
    states = tune([broken, fine], RandomSearch(), trials=96)
    assert len(states[0].measured) == 16  # its first round only
    assert len(states[1].measured) == 80


def test_validation_lets_the_default_win_when_noise_slowed_it() -> None:
    from tessera.tune.log import BestConfigs, Record
    from tessera.tune.tuner import validate

    task = SyntheticMatmul(16, 16, 16)
    default = task.space.default()
    base = task.cost(default)
    # A configuration that is really slower than the default, but faster than
    # the default's one measurement, which noise made 50% slower.
    other = next(p for p in task.space if base < task.cost(p) < 1.5 * base)
    log = TuningLog()
    log.append(Record(task.target_key, task.key, default, 1.5 * base))
    log.append(Record(task.target_key, task.key, other, task.cost(other)))
    before = BestConfigs(log.records()).lookup(task.target_key, task.key)
    assert before is not None
    assert before.config == other
    validate([task], runner=InlineRunner(), log=log, top=1, rounds=3)
    after = BestConfigs(log.records()).lookup(task.target_key, task.key)
    assert after is not None
    assert after.config == default
    per_config = Counter(config_json(r.config) for r in log.records())
    assert per_config == {config_json(default): 4, config_json(other): 4}


def test_validation_measures_top_configs_and_default_interleaved() -> None:
    from tessera.tune.tuner import validate

    task = SyntheticMatmul(16, 16, 16)
    log = TuningLog()
    tune([task], RandomSearch(), trials=20, log=log)
    order: list[int] = []
    validate(
        [task],
        runner=InlineRunner(),
        log=log,
        top=3,
        rounds=2,
        on_round=lambda r, t: order.append(r),
    )
    assert order == [0, 1]
    assert len(log.records()) == 20 + 2 * 4  # top 3 (the default among the 20) or 3 + default
