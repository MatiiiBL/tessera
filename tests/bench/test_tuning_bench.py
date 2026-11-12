"""The search-strategy comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tessera.bench.tuning import (
    Comparison,
    best_so_far,
    checkpoints,
    compare,
    table,
    trials_to_reach,
)
from tessera.tune.search import RandomSearch, TaskState
from tessera.tune.synthetic import SyntheticMatmul


def test_best_so_far_skips_failures() -> None:
    state = TaskState(SyntheticMatmul(4, 4, 4))
    space = state.task.space
    for i, s in enumerate([None, 3.0, 4.0, 1.0]):
        state.record(space.get(i), s)
    assert best_so_far(state) == [np.inf, 3.0, 3.0, 1.0]


def test_compare_runs_every_strategy_and_seed(tmp_path: Path) -> None:
    task = SyntheticMatmul(16, 16, 16)
    seen: list[tuple[str, str, int]] = []
    [c] = compare(
        [task],
        trials=24,
        seeds=[0, 1],
        optimum=lambda t: task.optimum()[1],
        log_dir=tmp_path,
        on_run=lambda *a: seen.append(a),
    )
    assert c.reference_kind == "optimum"
    assert seen == [(task.key, s, k) for s in ("random", "model") for k in (0, 1)]
    for strategy in ("random", "model"):
        rel = c.relative(strategy)
        assert rel.shape == (2, 24)
        assert np.all(np.diff(rel, axis=1) >= 0)
        assert np.all(rel <= 1.0 + 1e-12)
    assert len(list(tmp_path.glob("*.jsonl"))) == 4
    text = table([c])
    assert "| 24 |" in text
    assert "reference: optimum" in text


def test_reference_is_the_best_found_without_an_optimum() -> None:
    task = SyntheticMatmul(8, 8, 8)
    [c] = compare([task], trials=10, seeds=[0], strategies=["random"])
    assert c.reference_kind == "best found"
    assert c.relative("random")[0, -1] == 1.0
    assert c.as_dict()["runs"]["random"][0][-1] == c.reference


def test_checkpoints_and_trials_to_reach() -> None:
    assert checkpoints(100) == [8, 16, 32, 64, 100]
    c = Comparison("t", 1.0, "optimum", {"a": [[4.0, 2.0, 1.0], [2.0, 1.0, 1.0]]})
    assert trials_to_reach(c, "a", 0.9) == 2.5
    assert trials_to_reach(Comparison("t", 1.0, "optimum", {"a": [[2.0]]}), "a", 0.9) is None
    assert RandomSearch().name == "random"


def test_heaviest_tasks_rank_by_profiled_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import onnx

    from e2e.onnx_models import mlp
    from tessera.backend.c import CBackend
    from tessera.bench.tuning import heaviest_tasks
    from tessera.tune.space import Config

    def template(sch: Any, lowered: Any, cfg: Config) -> None:
        loop = sch.get_loops(lowered.main_block)[0]
        sch.split(loop, list(cfg.split("tile", sch.extent(loop), parts=2)))

    monkeypatch.setattr(CBackend, "templates", lambda self, target=None: {"matmul": template})
    path = tmp_path / "mlp.onnx"
    onnx.save(mlp(sizes=(64, 512, 64, 10)), path)
    tasks = heaviest_tasks(path, "c:generic", 2)
    assert len(tasks) == 2
    assert all(t.key.startswith("matmul:") for t in tasks)
    dims = [t.lowered.dims["N"] * t.lowered.dims["K"] for t in tasks]
    assert dims == sorted(dims, reverse=True)  # the 64x512 layers outweigh 64x10


def test_replay_of_an_exhaustively_measured_space(tmp_path: Path) -> None:
    from tessera.bench.tuning import Replay, measure_space
    from tessera.tune.log import TuningLog
    from tessera.tune.measure import InlineRunner
    from tessera.tune.space import config_json

    task = SyntheticMatmul(8, 8, 4)
    log = TuningLog(tmp_path / "space.jsonl")
    table = measure_space(task, InlineRunner(), log)
    assert len(table) == task.space.size
    assert len(log.records()) == task.space.size
    assert measure_space(task, InlineRunner(), log) == table  # resumed, nothing new
    replay = Replay(task, table)
    best, seconds = task.optimum()
    assert replay.optimum() == seconds
    assert replay.measure(replay.build(best)) == seconds
    assert replay.key == task.key
    [c] = compare([replay], trials=12, seeds=[0, 1], optimum=lambda t: replay.optimum())
    assert c.reference == seconds
    failed = Replay(task, {**table, config_json(best): None})
    with pytest.raises(RuntimeError, match="failed"):
        failed.measure(config_json(best))
