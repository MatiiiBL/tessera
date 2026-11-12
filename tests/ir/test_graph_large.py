"""Graphs far deeper than Python's recursion limit go through the whole pipeline."""

from __future__ import annotations

import numpy as np

from tessera.dtype import DType
from tessera.ir import Graph, TensorType, interpret
from tessera.passes import fuse, optimize, plan_memory, verify_groups

DEPTH = 12_000


def chain(depth: int) -> Graph:
    """x -> Softmax -> Relu -> Softmax -> ... (half anchors, half epilogues)."""
    g = Graph("deep")
    v = g.add_input("x", TensorType((4,), DType.float32))
    for i in range(depth):
        v = g.op("Softmax" if i % 2 == 0 else "Relu", v)
    g.set_outputs([v])
    return g


def test_deep_chain_runs_through_every_pass() -> None:
    g = chain(DEPTH)
    copied = g.copy()
    copied.verify()
    assert len(copied.nodes) == DEPTH
    assert copied.nodes[-1].output.name == g.nodes[-1].output.name
    assert copied.nodes[0].inputs[0] is copied.inputs[0]

    copied.nodes.reverse()
    copied.topo_sort()
    copied.verify()

    optimized = optimize(g)
    assert len(optimized.nodes) == DEPTH
    groups = fuse(optimized)
    verify_groups(optimized, groups)
    assert len(groups) == DEPTH // 2
    plan = plan_memory(optimized, groups)
    assert plan.arena_bytes == 2 * plan.alignment

    x = np.array([1.0, 2.0, 3.0, 4.0], np.float32)
    out = interpret(optimized, {"x": x})[optimized.outputs[0].name]
    np.testing.assert_allclose(out.sum(), 1.0, rtol=1e-6)


def test_copy_shares_constants_and_keeps_names() -> None:
    g = Graph("small")
    x = g.add_input("x", TensorType((3,), DType.float32))
    c = g.add_constant(np.ones(3, np.float32), "c")
    g.set_outputs([g.op("Add", x, c, name="sum")])
    copied = g.copy()
    assert copied.constants()[0].data is c.data
    assert copied.constants()[0] is not c
    assert [v.name for v in copied.values()] == [v.name for v in g.values()]
    assert copied.fresh_name("sum") == "sum_1"
