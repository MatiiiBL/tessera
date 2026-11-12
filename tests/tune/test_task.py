"""Tasks built from lowered groups: spaces from templates, keys and deduplication."""

from __future__ import annotations

import pytest

import tessera.tir
from tessera.dtype import DType
from tessera.ir.graph import Node, TensorType, Value
from tessera.plan import FusedGroup, LoweredGroup
from tessera.target import Target
from tessera.tir.schedule import Schedule
from tessera.tir.stmt import TRUE, Block, BlockRealize, Buffer, PrimFunc, SeqStmt
from tessera.tune.space import Config
from tessera.tune.task import Task, extract_tasks, task_key


def lowered(name: str, m: int) -> LoweredGroup:
    f32 = DType.float32
    a = Value(f"{name}_a", TensorType((m, 8), f32))
    b = Value(f"{name}_b", TensorType((8, 4), f32))
    out = Value(name, TensorType((m, 4), f32))
    node = Node("MatMul", [a, b], [out], name=name)
    group = FusedGroup(name, [node], [a, b], out, node, "matmul")
    params = (Buffer("A", (m, 8)), Buffer("B", (8, 4)), Buffer("C", (m, 4)))
    func = PrimFunc(name, params, BlockRealize((), TRUE, Block("root", (), (), (), SeqStmt(()))))
    return LoweredGroup(group, func, "C", None, {"B": 1, "M": m, "N": 4, "K": 8})


def template(sch: Schedule, group: LoweredGroup, cfg: Config) -> None:
    cfg.split("tile_m", group.dims["M"], parts=2, default=(group.dims["M"], 1))
    cfg.choice("vectorize", [False, True], default=True)


def test_space_comes_from_the_template() -> None:
    task = Task.create(lowered("mm", 16), template, Target(), key="matmul:test")
    assert task.space.size == 5 * 2
    assert task.space.default() == {"tile_m": (16, 1), "vectorize": True}
    assert task.target_key == Target().key
    func = task.schedule(task.space.get(3))
    assert func.name == "mm"


def test_identical_structures_share_a_task() -> None:
    groups = [lowered("a", 16), lowered("b", 32), lowered("c", 16)]
    tasks = extract_tasks(
        groups, {"matmul": template}, Target(), key=lambda g: f"matmul:{g.dims['M']}"
    )
    assert [t.key for t in tasks] == ["matmul:16", "matmul:32"]
    assert [t.weight for t in tasks] == [2, 1]
    assert tasks[0].names == ["a", "c"]


def test_task_key_uses_the_structural_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(tessera.tir.__dict__, "structural_hash", lambda f: "0123456789abcdef99")
    assert task_key(lowered("x", 4)) == "matmul:0123456789abcdef"
