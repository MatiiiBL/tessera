"""Compaction of local and shared buffers to what one thread or threadgroup touches."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from metal_funcs import assert_close, bound, gpu, loop, matmul_tiled, prim_func, run_case

from tessera.backend.metal.codegen import generate, intrinsic_extents
from tessera.backend.metal.lower import compact_buffers, interval, plan_compaction, terms
from tessera.tir.expr import FloatImm, Var
from tessera.tir.stmt import Buffer, BufferStore, seq


def shapes(func: object) -> dict[str, tuple[int, ...]]:
    plan, _ = plan_compaction(func, intrinsic_extents)  # type: ignore[arg-type]
    return {c.old.name: c.new.shape for c in plan.values()}


def test_cache_buffers_shrink_to_one_thread_and_one_threadgroup() -> None:
    case = matmul_tiled(64, 64, 32, np.random.default_rng(0), threads=8, reg=2, kt=8)
    # A threadgroup computes a 16 x 16 tile, 8 columns of K at a time; a thread
    # keeps a 2 x 2 tile of accumulators.
    assert shapes(case.func) == {"A_shared": (16, 8), "B_shared": (8, 16), "C_local": (2, 2)}
    kernel = generate(case.func)
    assert kernel.threadgroup_memory == 2 * 16 * 8 * 4
    assert "threadgroup float A_shared[128];" in kernel.source
    assert "float C_local[4];" in kernel.source
    assert "C_local[i * 2 + j] = 0.0f;" in kernel.source
    assert "A_shared[e * 64 + thread_y * 8 + thread_x] = A[" in kernel.source


def test_local_buffer_used_by_one_thread_loop_keeps_the_thread_fixed() -> None:
    out = Buffer("out", (64,))
    tmp = Buffer("tmp", (64, 4), scope="local")
    t, j, j2 = Var("t"), Var("j"), Var("j2")
    body = bound(
        t,
        64,
        "thread.x",
        seq(
            [
                loop(j, 4, BufferStore(tmp, FloatImm(1.0), (t, j))),
                loop(j2, 4, BufferStore(out, out[t] + tmp[t, j2], (t,))),
            ]
        ),
    )
    func = prim_func("per_thread", [out], body, alloc=[tmp])
    assert shapes(func) == {"tmp": (1, 4)}
    source = generate(func).source
    assert "float tmp[4];" in source
    assert "tmp[j] = 1.0f;" in source


def test_shared_buffer_spans_every_thread_of_the_group() -> None:
    out = Buffer("out", (4, 32))
    sh = Buffer("sh", (4, 32), scope="shared")
    g, t, t2 = Var("g"), Var("t"), Var("t2")
    body = bound(
        g,
        4,
        "threadgroup.x",
        seq(
            [
                bound(t, 32, "thread.x", BufferStore(sh, FloatImm(2.0), (g, t))),
                bound(t2, 32, "thread.x", BufferStore(out, sh[g, 31 - t2], (g, t2))),
            ]
        ),
    )
    func = prim_func("group", [out], body, alloc=[sh])
    assert shapes(func) == {"sh": (1, 32)}
    assert "sh[31 - thread_x]" in generate(func).source


def test_unknown_accesses_keep_the_full_shape() -> None:
    out = Buffer("out", (8,))
    tmp = Buffer("tmp", (8,), scope="local")
    idx = Buffer("idx", (8,), out.dtype)
    i = Var("i")
    from tessera.dtype import DType
    from tessera.tir.expr import Cast

    body = loop(
        i,
        8,
        seq(
            [
                BufferStore(tmp, FloatImm(1.0), (i,)),
                BufferStore(out, tmp[Cast(DType.int64, idx[i])], (i,)),
            ]
        ),
    )
    func = prim_func("gather_local", [idx, out], body, alloc=[tmp])
    assert shapes(func) == {}
    assert compact_buffers(func, intrinsic_extents) is func


def test_terms_and_intervals() -> None:
    a, b = Var("a"), Var("b")
    assert [s for s, _ in terms(a + b - (a * 2) + 3)] == [1, 1, -1, 1]
    ranges = {a: (0, 9), b: (2, 5)}
    assert interval(a // 4, ranges) == (0, 2)
    assert interval(a % 4, ranges) == (0, 3)
    assert interval(b % 8, ranges) == (2, 5)
    assert interval(a * b - b, ranges) == (-5, 43)


@gpu
@given(
    threads=st.sampled_from([2, 4, 8]),
    reg=st.sampled_from([1, 2, 4]),
    kt=st.sampled_from([4, 8]),
    tiles=st.integers(1, 3),
)
@settings(max_examples=15, deadline=None)
def test_compacted_matmul_is_correct(threads: int, reg: int, kt: int, tiles: int) -> None:
    bm = threads * reg
    if (bm * kt) % (threads * threads):
        return
    case = matmul_tiled(
        bm * tiles, bm * 2, kt * 3, np.random.default_rng(threads), threads=threads, reg=reg, kt=kt
    )
    result, reference = run_case(case)
    assert_close(result, reference, rtol=1e-4, atol=1e-4)


def test_compaction_respects_the_threadgroup_memory_limit() -> None:
    from tessera.backend.metal.codegen import CodegenError

    case = matmul_tiled(512, 512, 512, np.random.default_rng(0), threads=8, reg=2, kt=8)
    generate(case.func)  # fits once compacted
    with pytest.raises(CodegenError, match="threadgroup memory"):
        generate(case.func, max_threadgroup_memory=1000)
