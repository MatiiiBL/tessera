"""Memory planning: storage kinds, aliasing, arena reuse and lifetimes."""

from __future__ import annotations

import numpy as np
import pytest
from graph_strategies import graphs, run_plan
from hypothesis import HealthCheck, given, settings

from tessera.dtype import DType
from tessera.ir import Graph, TensorType, interpret, random_inputs
from tessera.passes import FusionOptions, fuse, plan_memory
from tessera.passes.memory import _Arena
from tessera.plan import FusedGroup, MemoryPlan, StorageKind

F32 = DType.float32


def chain(n: int, size: int = 64) -> Graph:
    """x -> Softmax -> Softmax -> ... (each kernel reads only the previous result)."""
    g = Graph("chain")
    v = g.add_input("x", TensorType((size,), F32))
    for i in range(n):
        v = g.op("Softmax", v, name=f"s{i}")
    g.set_outputs([v])
    return g


def lifetimes(groups: list[FusedGroup]) -> dict[str, tuple[int, int]]:
    spans: dict[str, tuple[int, int]] = {}
    for pos, g in enumerate(groups):
        spans.setdefault(g.output.name, (pos, pos))
        for v in g.inputs:
            if v.name in spans:
                spans[v.name] = (spans[v.name][0], pos)
    return spans


def check_plan(g: Graph, groups: list[FusedGroup], plan: MemoryPlan) -> None:
    """No two arena values that are live at once share a byte; refs are aligned."""
    spans = lifetimes(groups)
    arena = [(n, r) for n, r in plan.refs.items() if r.kind is StorageKind.ARENA]
    for name, ref in arena:
        assert ref.offset % plan.alignment == 0
        assert ref.offset + ref.nbytes <= plan.arena_bytes
        assert name in spans
    for i, (a, ra) in enumerate(arena):
        for b, rb in arena[i + 1 :]:
            if ra.offset == rb.offset and ra.nbytes == rb.nbytes:
                continue  # a view and its input
            overlap_bytes = ra.offset < rb.offset + rb.nbytes and rb.offset < ra.offset + ra.nbytes
            (sa, ea), (sb, eb) = spans[a], spans[b]
            assert not (overlap_bytes and sa <= eb and sb <= ea), (a, b)
    feeds = random_inputs(g)
    expected = interpret(g, feeds)
    got = run_plan(g, groups, plan, feeds)
    for name, value in expected.items():
        np.testing.assert_allclose(got[name], value, rtol=1e-5, atol=1e-6)


def test_chain_reuses_two_buffers() -> None:
    g = chain(6)
    groups = fuse(g)
    plan = plan_memory(g, groups)
    check_plan(g, groups, plan)
    offsets = {r.offset for r in plan.refs.values() if r.kind is StorageKind.ARENA}
    assert offsets == {0, 256}
    assert plan.arena_bytes == 512
    assert plan.refs["x"].kind is StorageKind.INPUT
    assert plan.refs["s5"].kind is StorageKind.OUTPUT


def test_alignment_rounds_offsets() -> None:
    g = chain(3, size=3)
    plan = plan_memory(g, fuse(g), alignment=64)
    assert {r.offset for r in plan.refs.values() if r.kind is StorageKind.ARENA} == {0, 64}
    with pytest.raises(ValueError, match="power of two"):
        plan_memory(g, fuse(g), alignment=48)


def test_constants_are_packed_into_one_blob() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((2, 3), F32))
    w = g.add_constant(np.arange(12, dtype=np.float32).reshape(3, 4), "w")
    idx = g.add_constant(np.array([1, 0], np.int64), "idx")
    flag = g.add_constant(np.array([True, False, True, True]), "flag")
    y = g.op("Gather", g.op("MatMul", x, w), idx)
    g.set_outputs([g.op("Where", flag, y, g.add_constant(np.float32(0), "zero"))])
    groups = fuse(g)
    plan = plan_memory(g, groups)
    for v in g.constants():
        ref = plan.refs[v.name]
        assert ref.kind is StorageKind.CONSTANT
        assert ref.offset % 256 == 0
        assert v.data is not None
        stored = plan.constants[ref.offset : ref.offset + ref.nbytes].view(v.data.dtype)
        np.testing.assert_array_equal(stored.reshape(v.shape), v.data)
    check_plan(g, groups, plan)


def test_view_shares_its_input_storage() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((2, 6), F32))
    s = g.op("Softmax", x, name="s")
    r = g.op("Reshape", s, shape=(3, 4), name="r")
    g.set_outputs([g.op("Softmax", r, name="t")])
    groups = fuse(g, options=FusionOptions(prologue=False))
    assert [gr.is_view for gr in groups] == [False, True, False]
    plan = plan_memory(g, groups)
    assert plan.refs["r"] == plan.refs["s"]
    assert plan.refs["r"].kind is StorageKind.ARENA
    check_plan(g, groups, plan)


def test_kernel_writes_straight_into_an_output_that_is_its_view() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((2, 6), F32))
    s = g.op("Softmax", x, name="s")
    g.set_outputs([g.op("Reshape", s, shape=(12,), name="flat")])
    groups = fuse(g)
    plan = plan_memory(g, groups)
    assert plan.refs["s"] == plan.refs["flat"]
    assert plan.refs["flat"].kind is StorageKind.OUTPUT
    assert plan.arena_bytes == 0
    check_plan(g, groups, plan)


def test_views_of_inputs_and_outputs_returned_as_outputs() -> None:
    g = Graph()
    x = g.add_input("x", TensorType((2, 3), F32))
    s = g.op("Softmax", x, name="s")
    g.set_outputs([s, g.op("Reshape", x, shape=(6,), name="xv"), g.op("Reshape", s, shape=(6,))])
    groups = fuse(g)
    plan = plan_memory(g, groups)
    assert plan.refs["xv"].kind is StorageKind.OUTPUT
    assert plan.refs["xv"].index == 1
    check_plan(g, groups, plan)


def test_arena_is_smaller_than_the_sum_of_intermediates() -> None:
    g = chain(10, size=1000)
    groups = fuse(g, enabled=False)
    plan = plan_memory(g, groups)
    total = sum(r.nbytes for r in plan.refs.values() if r.kind is StorageKind.ARENA)
    assert plan.arena_bytes * 4 < total


def test_arena_best_fit_reuses_the_tightest_hole() -> None:
    arena = _Arena(1)
    a, b, c = arena.allocate(100), arena.allocate(10), arena.allocate(30)
    arena.release(a, 100)
    arena.release(c, 30)
    assert arena.allocate(25) == c
    assert arena.allocate(90) == a
    arena.release(b, 10)
    assert arena.free == [(90, 20), (135, 5)]
    assert arena.size == 140


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(graphs())
def test_plans_are_sound_on_random_graphs(g: Graph) -> None:
    for enabled in (True, False):
        groups = fuse(g, enabled=enabled)
        check_plan(g, groups, plan_memory(g, groups))
