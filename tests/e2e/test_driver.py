"""The driver: plans, copies, tuning-log lookup and template application."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from e2e.onnx_models import views_model
from tessera import te
from tessera.backend.base import Template
from tessera.backend.metal.runtime import available as metal_available
from tessera.driver import (
    Kernel,
    apply_template,
    best_configs,
    copy_kernel,
    default_logs,
    lower,
    resolve_target,
    schedule_kernel,
)
from tessera.dtype import DType
from tessera.ir.graph import Node, TensorType, Value
from tessera.plan import BufferRef, FusedGroup, LoweredGroup, StorageKind
from tessera.target import Target
from tessera.tir import interpret
from tessera.tir.schedule import Schedule, ScheduleError
from tessera.tir.stmt import Buffer
from tessera.tune.log import BestConfigs, Record, TuningLog
from tessera.tune.space import Config
from tessera.tune.task import Task

GENERIC = Target(kind="c", isa="generic")


@pytest.mark.parametrize(
    "target",
    [
        "c:generic",
        pytest.param(
            "metal",
            marks=[
                pytest.mark.metal,
                pytest.mark.skipif(not metal_available(), reason="no Metal device"),
            ],
        ),
    ],
)
def test_outputs_that_are_views_are_copied(target: str) -> None:
    comp = lower(views_model(), target=target)
    assert [g.is_view for g in comp.groups] == [True, True]
    assert [k.name for k in comp.kernels] == ["copy_y", "copy_z"]
    for k, index in zip(comp.kernels, (0, 1), strict=True):
        src, dst = k.args
        assert src.kind is StorageKind.INPUT
        assert dst == BufferRef(StorageKind.OUTPUT, 0, 96, index)
    assert comp.memory.refs["y"].kind is StorageKind.OUTPUT
    x = np.random.default_rng(0).standard_normal((2, 3, 4)).astype(np.float32)
    out = comp.build().run({"x": x})
    assert np.array_equal(out["y"], x.reshape(6, 4))
    assert np.array_equal(out["z"], x)


def test_copy_kernel() -> None:
    src, dst = Buffer("a", (10,)), Buffer("b", (10,))
    func = copy_kernel("copy_b", src, dst)
    a, b = np.arange(10, dtype=np.float32), np.zeros(10, np.float32)
    interpret(func, [a, b])
    assert np.array_equal(a, b)
    assert func.params == (src, dst)


def test_targets_and_default_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert resolve_target("c:generic") == GENERIC
    assert resolve_target(GENERIC) is GENERIC
    assert resolve_target("metal").kind == "metal"
    assert default_logs().parts[-2:] == ("results", "tuning")
    monkeypatch.setenv("TESSERA_TUNING_LOGS", str(tmp_path))
    assert default_logs() == tmp_path


def test_best_configs_accepts_files_directories_and_lists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    TuningLog(tmp_path / "a.jsonl").append(Record("t", "a", {"x": 1}, 1.0))
    TuningLog(tmp_path / "b" / "b.jsonl").append(Record("t", "b", {"x": 2}, 1.0))
    assert len(best_configs(tmp_path)) == 2
    assert len(best_configs(tmp_path / "a.jsonl")) == 1
    assert len(best_configs([tmp_path / "a.jsonl", tmp_path / "b"])) == 2
    given = BestConfigs()
    assert best_configs(given) is given
    monkeypatch.setenv("TESSERA_TUNING_LOGS", str(tmp_path / "b"))
    assert len(best_configs(None)) == 1


# -- template application on te-built functions ----------------------------------


def matmul_group(name: str = "mm", m: int = 16, k: int = 8, n: int = 12) -> LoweredGroup:
    a = te.placeholder((m, k), name="A")
    b = te.placeholder((k, n), name="B")
    r = te.reduce_axis(k, "k")
    c = te.compute((m, n), lambda i, j: te.sum(a[i, r] * b[r, j], r), name="C")
    func = te.create_prim_func([a, b, c], name=name)
    f32 = DType.float32
    va, vb = (
        Value(f"{name}_a", TensorType((m, k), f32)),
        Value(f"{name}_b", TensorType((k, n), f32)),
    )
    out = Value(name, TensorType((m, n), f32))
    node = Node("MatMul", [va, vb], [out], name=name)
    group = FusedGroup(name, [node], [va, vb], out, node, "matmul")
    return LoweredGroup(group, func, "C", None, {"B": 1, "M": m, "N": n, "K": k})


def template(sch: Schedule, lowered: LoweredGroup, cfg: Config) -> None:
    i = sch.get_loops("C")[0]
    outer, _ = sch.split(i, list(cfg.split("tile_i", lowered.dims["M"], parts=2)))
    if cfg.choice("parallel", [False, True], default=True):
        sch.parallel(outer)


def strict_template(sch: Schedule, lowered: LoweredGroup, cfg: Config) -> None:
    """Rejects one point of its own space, as real templates may."""
    if cfg.split("tile_i", lowered.dims["M"], parts=2) == (2, 8):
        raise ScheduleError("tile rejected")
    cfg.choice("parallel", [False, True])


def schedule(
    lowered: LoweredGroup, best: BestConfigs, tmpl: Template = template
) -> tuple[Task, Kernel]:
    task = Task.create(lowered, tmpl, GENERIC)
    args = tuple(BufferRef(StorageKind.ARENA, 0, 4) for _ in range(3))
    return task, schedule_kernel(lowered.group, lowered, args, tmpl, task, best, GENERIC)


def test_defaults_without_a_log() -> None:
    lowered = matmul_group()
    task, kernel = schedule(lowered, BestConfigs())
    assert not kernel.tuned
    assert kernel.config == task.space.default()
    assert kernel.name == "mm"
    assert kernel.trace[-1].startswith("parallel")


def test_the_fastest_valid_logged_config_wins() -> None:
    lowered = matmul_group()
    task, _ = schedule(lowered, BestConfigs())
    key = GENERIC.key
    records = [
        Record(key, task.key, {"tile_i": (4, 4), "parallel": False}, 2.0),
        Record(key, task.key, {"tile_i": (8, 2), "parallel": True}, 1.0),
        Record(key, task.key, {"tile_i": (3, 5), "parallel": True}, 0.5),  # not in the space
        Record(key, task.key, {"tile_i": (2, 8), "parallel": True}, None, "build: crash"),
        Record("other-target", task.key, {"tile_i": (16, 1), "parallel": True}, 0.1),
    ]
    _, kernel = schedule(lowered, BestConfigs(records))
    assert kernel.tuned
    assert kernel.config == {"tile_i": (8, 2), "parallel": True}
    a = np.random.default_rng(0).random((16, 8), dtype=np.float32)
    b = np.random.default_rng(1).random((8, 12), dtype=np.float32)
    c = np.zeros((16, 12), np.float32)
    interpret(kernel.func, [a, b, c])
    np.testing.assert_allclose(c, a @ b, rtol=1e-5)


def test_a_logged_config_the_template_rejects_falls_back_to_defaults() -> None:
    lowered = matmul_group()
    task, _ = schedule(lowered, BestConfigs(), strict_template)
    record = Record(GENERIC.key, task.key, {"tile_i": (2, 8), "parallel": True}, 1.0)
    _, kernel = schedule(lowered, BestConfigs([record]), strict_template)
    assert not kernel.tuned
    assert kernel.config == task.space.default()


def test_apply_template_returns_the_trace() -> None:
    func, trace = apply_template(template, matmul_group(), {"tile_i": (4, 4), "parallel": True})
    assert trace == ["split(i0, [4, 4]) -> [i0_0, i0_1]", "parallel(i0_0)"]
    assert func.name == "mm"
