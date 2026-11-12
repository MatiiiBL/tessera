"""The process runner: parallel builds, isolated measurements, failures recorded."""

from __future__ import annotations

import os
import signal
import sys
import time
import types
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest

from tessera.tune.measure import InlineRunner, LocalRunner
from tessera.tune.space import KnobValue
from tessera.tune.synthetic import SyntheticMatmul


@dataclass(frozen=True)
class Flaky(SyntheticMatmul):
    """Misbehaves on purpose, keyed by the first factor of ``tile_m``."""

    def build(self, config: Mapping[str, KnobValue]) -> dict[str, KnobValue]:
        mo = config["tile_m"][0]  # type: ignore[index]
        if mo == 1:
            os.kill(os.getpid(), signal.SIGSEGV)
        if mo == 2:
            raise ValueError("bad tile")
        if mo == 4:
            time.sleep(30)
        return dict(config)

    def measure(self, built: Mapping[str, KnobValue]) -> float:
        if built["tile_m"] == (8, 1, 1):
            os._exit(3)
        if built["tile_m"] == (8, 2, 1):
            time.sleep(30)
        return self.cost(built)


def points(task: SyntheticMatmul, tile_ms: list[tuple[int, ...]]) -> list[dict[str, KnobValue]]:
    base = task.space.default()
    return [{**base, "tile_m": t} for t in tile_ms]


def test_local_runner_matches_inline_runner() -> None:
    task = SyntheticMatmul(16, 16, 16)
    configs = [task.space.get(i) for i in range(0, task.space.size, task.space.size // 7)]
    with LocalRunner(build_workers=2) as runner:
        local = runner.run(task, configs)
        again = runner.run(task, configs[:2])  # workers keep running between batches
    inline = InlineRunner().run(task, configs)
    assert local == inline
    assert runner.build_seconds > 0
    assert runner.measure_seconds > 0
    assert again == inline[:2]


def test_crashes_timeouts_and_errors_are_isolated() -> None:
    task = Flaky(16, 16, 16)
    tiles = [(16, 1, 1), (1, 16, 1), (2, 8, 1), (4, 4, 1), (8, 1, 2), (8, 1, 1), (8, 2, 1)]
    configs = points(task, tiles)
    configs.append({**configs[0], "vectorize": True})
    with LocalRunner(build_workers=3, build_timeout=2.0, measure_timeout=2.0) as runner:
        results = runner.run(task, configs)
    errors = [r.error for r in results]
    assert results[0].seconds == task.cost(configs[0])
    assert errors[1] == "build: process killed by SIGSEGV"
    assert errors[2] == "build: ValueError: bad tile"
    assert errors[3] == "build: timed out after 2 s"
    assert results[4].seconds is not None
    assert errors[5] == "measure: process exited with code 3"
    assert errors[6] == "measure: timed out after 2 s"
    assert results[7].seconds == task.cost(configs[7])  # the run went on
    assert [r.config for r in results] == configs


def test_unpicklable_tasks_fail_their_candidates() -> None:
    @dataclass(frozen=True)
    class Local(SyntheticMatmul):  # defined in a function: cannot be pickled
        pass

    task = Local(4, 4, 4)
    with LocalRunner(build_workers=1) as runner:
        results = runner.run(task, [task.space.default()])
    assert results[0].seconds is None
    assert "cannot send to the build process" in (results[0].error or "")


def test_needs_a_build_process() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        LocalRunner(build_workers=0)


def test_tasks_the_worker_cannot_import_fail_their_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("only_in_this_process")

    @dataclass(frozen=True)
    class Elsewhere(SyntheticMatmul):
        pass

    Elsewhere.__module__ = module.__name__
    Elsewhere.__qualname__ = "Elsewhere"
    module.Elsewhere = Elsewhere  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    task = Elsewhere(4, 4, 4)
    with LocalRunner(build_workers=1) as runner:
        results = runner.run(task, [task.space.default()] * 2)
        other = SyntheticMatmul(8, 8, 8)  # a different key: the worker loads it
        fine = runner.run(other, [other.space.default()])
    assert all("cannot load the task here" in (r.error or "") for r in results)
    assert fine[0].seconds is not None


def test_slow_worker_start_does_not_count_against_timeouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Workers get the tuner's sys.path; this makes each of them start 2.5 s late.
    (tmp_path / "sitecustomize.py").write_text("import time\ntime.sleep(2.5)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    task = SyntheticMatmul(4, 4, 4)
    with LocalRunner(build_workers=1, build_timeout=1.0, measure_timeout=1.0) as runner:
        [result] = runner.run(task, [task.space.default()])
    assert result.error is None
    assert result.seconds == task.cost(task.space.default())
