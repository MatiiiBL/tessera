"""Lowering to opaque blocks and compacting local buffers keep the result."""

from __future__ import annotations

import numpy as np

from tessera import tir
from tessera.tir.primitives.common import iter_paths
from tessera.tir.schedule import Schedule
from tessera.tir.stmt import Block, BlockRealize, Buffer, For, PrimFunc

from .workloads import Workload, box_filter, matmul_bias_relu, run


def allocations(func: PrimFunc) -> dict[str, tuple[Buffer, object]]:
    out: dict[str, tuple[Buffer, object]] = {}
    for path in iter_paths(func.body):
        node = path[-1]
        if isinstance(node, Block):
            owner = next((n.loop_var.name for n in reversed(path) if isinstance(n, For)), None)
            for b in node.alloc_buffers:
                out[b.name] = (b, owner)
    return out


def tiled_matmul() -> tuple[Schedule, Workload]:
    w = matmul_bias_relu(8, 8, 8)
    s = Schedule(w.func)
    out = s.cache_write("matmul", 0, "local")
    i, j, k = s.get_loops("matmul")
    io, ii = s.split(i, [None, 4])
    jo, ji = s.split(j, [None, 4])
    s.reorder(io, jo, k, ii, ji)
    s.reverse_compute_at(out, jo)
    s.decompose_reduction("matmul", k)
    a = s.cache_read("matmul", 0, "local")
    s.compute_at(a, k)
    return s, w


def test_lower_blocks_removes_iteration_variables() -> None:
    s, w = tiled_matmul()
    lowered = tir.transform.lower_blocks(s.func)
    tir.verify(lowered)
    for path in iter_paths(lowered.body):
        if isinstance(path[-1], BlockRealize):
            assert not path[-1].block.iter_vars
            assert path[-1].block.init is None
    np.testing.assert_allclose(run(lowered, w), w.expected(), rtol=1e-5, atol=1e-5)


def test_compact_buffers_allocates_tiles_where_they_are_used() -> None:
    s, w = tiled_matmul()
    compact = tir.transform.compact_buffers(s.func)
    tir.verify(compact)
    allocs = allocations(compact)
    acc, acc_owner = allocs["matmul_local"]
    assert acc.shape == (4, 4)
    assert acc_owner == "j_0"
    a_tile, a_owner = allocs["A_local"]
    assert a_tile.shape == (4, 1)
    assert a_owner == "k"
    np.testing.assert_allclose(run(compact, w), w.expected(), rtol=1e-5, atol=1e-5)


def test_shared_buffers_stay_outside_thread_loops() -> None:
    w = box_filter(8, 4, 2)
    s = Schedule(w.func)
    h = s.get_loops("box")[0]
    ho, hi = s.split(h, [None, 4])
    s.bind(ho, "threadgroup.x")
    s.bind(hi, "thread.x")
    p = s.cache_read("box", 0, "shared")
    s.compute_at(p, hi)
    compact = tir.transform.compact_buffers(s.func)
    tir.verify(compact)
    buf, owner = allocations(compact)["pad_shared"]
    assert owner == "h_0"
    assert buf.shape == (6, 6, 2)
    np.testing.assert_allclose(run(compact, w), w.expected(), rtol=1e-5, atol=1e-5)
