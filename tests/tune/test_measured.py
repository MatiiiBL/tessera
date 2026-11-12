"""Tuning real loop nests on the C backend: processes, prebuild, measure, log."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tessera.target import Target
from tessera.tune.log import BestConfigs, TuningLog
from tessera.tune.measure import LocalRunner
from tessera.tune.search import EvolutionarySearch
from tessera.tune.space import config_json
from tessera.tune.synthetic import MeasuredMatmul, SyntheticMatmul
from tessera.tune.tuner import tune

pytestmark = pytest.mark.skipif(shutil.which("clang") is None, reason="needs clang")


def test_tuning_measures_real_kernels(tmp_path: Path) -> None:
    task = MeasuredMatmul(SyntheticMatmul(32, 32, 32), Target(kind="c", isa="generic"))
    log = TuningLog(tmp_path / "log.jsonl")
    with LocalRunner(build_workers=2) as runner:
        state = tune([task], EvolutionarySearch(), trials=12, batch_size=6, runner=runner, log=log)[
            0
        ]
    records = log.records()
    assert len(records) == 12
    assert all(r.ok for r in records), [r.error for r in records if not r.ok]
    assert all(0 < (r.seconds or 0) < 0.1 for r in records)
    assert records[0].target == Target(kind="c", isa="generic").key
    best = state.best()
    default = state.measured[config_json(task.space.default())]
    assert best is not None
    assert default is not None
    assert best[1] <= default
    found = BestConfigs.load(tmp_path).lookup(task.target_key, task.key)
    assert found is not None
    assert found.seconds == best[1]
