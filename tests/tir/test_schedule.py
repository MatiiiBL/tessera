"""Each schedule primitive: what it does, that it keeps the result, and when it refuses."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tessera import te, tir
from tessera.dtype import DType
from tessera.tir.expr import IntImm
from tessera.tir.schedule import Schedule, ScheduleError
from tessera.tir.stmt import Block, BlockRealize, For, ForKind

from .workloads import Workload, box_filter, elementwise_chain, matmul_bias_relu, row_softmax, run


def check(s: Schedule, w: Workload) -> None:
    tir.verify(s.func)
    np.testing.assert_allclose(run(s.func, w), w.expected(), rtol=1e-4, atol=1e-5)


def loops(s: Schedule, block: str) -> list[For]:
    out = []
    for v in s.get_loops(block):
        node = _find_for(s, v)
        out.append(node)
    return out


def _find_for(s: Schedule, v: object) -> For:
    from tessera.tir.primitives.common import find_loop, loop_of

    return loop_of(find_loop(s.func, v))  # type: ignore[arg-type]


def realize(s: Schedule, block: str) -> BlockRealize:
    from tessera.tir.primitives.common import find_block, realize_of

    return realize_of(find_block(s.func, block))


def refuses(s: Schedule, match: str, fn: object, *args: object) -> None:
    before = s.func
    with pytest.raises(ScheduleError, match=match):
        fn(*args)  # type: ignore[operator]
    assert s.func is before


# -- lookup ---------------------------------------------------------------------------


def test_lookup_and_copy() -> None:
    w = matmul_bias_relu(8, 6, 4)
    s = Schedule(w.func)
    assert s.blocks() == ["matmul", "relu"]
    i, j, k = s.get_loops("matmul")
    assert [s.extent(x) for x in (i, j, k)] == [8, 6, 4]
    c = s.copy()
    c.split(i, [2, 4])
    assert s.func is w.func
    assert c.func is not w.func
    assert c.trace == ["split(i, [2, 4]) -> [i_0, i_1]"]
    assert s.trace == []
    with pytest.raises(ScheduleError, match="no block"):
        s.get_loops("nope")


# -- split / fuse / reorder -------------------------------------------------------------


def test_split_perfect() -> None:
    w = matmul_bias_relu(8, 6, 4)
    s = Schedule(w.func)
    i = s.get_loops("matmul")[0]
    a, b, c = s.split(i, [2, None, 2])
    assert [s.extent(x) for x in (a, b, c)] == [2, 2, 2]
    assert realize(s, "matmul").predicate == IntImm(1, DType.bool_)
    check(s, w)


def test_split_imperfect_adds_predicate() -> None:
    w = matmul_bias_relu(7, 6, 5)
    s = Schedule(w.func)
    i, _, k = s.get_loops("matmul")
    io, _ii = s.split(i, [None, 3])
    assert s.extent(io) == 3
    assert "where(i_0 * 3 + i_1 < 7)" in tir.script(s.func)
    s.split(k, [2, 3])
    check(s, w)


def test_split_refusals() -> None:
    s = Schedule(matmul_bias_relu(8, 6, 4).func)
    i = s.get_loops("matmul")[0]
    refuses(s, "cover", s.split, i, [2, 2])
    refuses(s, "at most one", s.split, i, [None, None])
    refuses(s, "positive", s.split, i, [0, 8])
    s.parallel(i)
    refuses(s, "parallel", s.split, i, [2, 4])


def test_fuse_and_split_round_trip() -> None:
    w = elementwise_chain(6, 4)
    s = Schedule(w.func)
    i, j = s.get_loops("transpose")
    f = s.fuse(i, j)
    assert s.extent(f) == 24
    s.split(f, [None, 4])
    check(s, w)
    body = tir.script(s.func)
    assert "i_j_fused_0" in body
    assert "i_j_fused_1" in body


def test_fuse_refusals() -> None:
    s = Schedule(matmul_bias_relu(4, 4, 4).func)
    i, j, k = s.get_loops("matmul")
    refuses(s, "spatial and reduction", s.fuse, j, k)
    refuses(s, "consecutive", s.fuse, j, i)
    _i2, j2 = s.get_loops("relu")
    refuses(s, "nested", s.fuse, i, j2)


def test_reorder() -> None:
    w = matmul_bias_relu(4, 6, 5)
    s = Schedule(w.func)
    i, j, k = s.get_loops("matmul")
    s.reorder(k, i)
    assert s.get_loops("matmul") == [k, j, i]
    check(s, w)


def test_reorder_refusals() -> None:
    w = box_filter(4, 4, 2)
    s = Schedule(w.func)
    h = s.get_loops("box")[0]
    s.compute_at("pad", h)
    w_loop = s.get_loops("box")[1]
    ax = s.get_loops("pad")[1]
    refuses(s, "perfectly nested|not nested", s.reorder, ax, w_loop)
    refuses(s, "twice", s.reorder, h, h)


# -- loop kinds --------------------------------------------------------------------------


def test_parallel_vectorize_unroll() -> None:
    w = matmul_bias_relu(8, 8, 4)
    s = Schedule(w.func)
    i, _j, k = s.get_loops("matmul")
    s.parallel(i)
    s.unroll(k)
    _jo, ji = s.split(s.get_loops("relu")[1], [None, 4])
    s.vectorize(ji)
    kinds = [f.kind for f in loops(s, "matmul")]
    assert kinds == [ForKind.PARALLEL, ForKind.SERIAL, ForKind.UNROLLED]
    assert loops(s, "relu")[-1].kind == ForKind.VECTORIZED
    check(s, w)


def test_parallel_refuses_reduction_and_shared_writes() -> None:
    w = matmul_bias_relu(8, 8, 4)
    s = Schedule(w.func)
    _, j, k = s.get_loops("matmul")
    refuses(s, "reduction loop", s.parallel, k)
    refuses(s, "has loops inside", s.vectorize, j)
    s.parallel(j)
    refuses(s, "already", s.vectorize, j)


def test_parallel_over_a_privatizable_intermediate() -> None:
    w = box_filter(6, 4, 2)
    s = Schedule(w.func)
    ho, _ = s.split(s.get_loops("box")[0], [None, 2])
    s.compute_at("pad", ho)
    s.parallel(ho)
    check(s, w)


def test_bind_and_cross_thread_reduction() -> None:
    w = row_softmax(4, 8)
    s = Schedule(w.func)
    i, k = s.get_loops("rowsum")
    s.bind(i, "threadgroup.x")
    s.bind(k, "thread.x")
    refuses(s, "reduction loop", s.bind, s.get_loops("rowmax")[1], "threadgroup.y")
    refuses(s, "unknown thread axis", s.bind, s.get_loops("rowmax")[0], "warp.x")
    assert loops(s, "rowsum")[1].thread_tag == "thread.x"
    check(s, w)


def test_local_buffers_under_different_thread_axes_fail_verification() -> None:
    w = box_filter(2, 2, 2)
    s = Schedule(w.func)
    s.cache_write("pad", 0, "local")
    fused = s.fuse(*s.get_loops("pad"))
    s.bind(fused, "thread.x")  # the copy-out still runs outside the thread loop
    tir.verify(s.func, threads=False)
    with pytest.raises(tir.VerifyError, match="different thread axes"):
        tir.verify(s.func)
    with pytest.raises(tir.VerifyError, match="each thread sees only its own copy"):
        run(s.func, w)


def gpu_matmul(m: int, n: int, k: int, tile: int) -> tuple[Schedule, Workload]:
    """The tiled GPU matmul: a register accumulator per thread, initialized, updated
    and stored by three nests that each bind thread.y and thread.x."""
    a = te.placeholder((m, k), DType.float32, "A")
    b = te.placeholder((k, n), DType.float32, "B")
    r = te.reduce_axis(k, "k")
    c = te.compute((m, n), lambda i, j: te.sum(a[i, r] * b[r, j], axis=r), "C", ["i", "j"])
    rng = np.random.default_rng(3)
    w = Workload(
        te.create_prim_func([a, b, c]),
        [
            rng.standard_normal((m, k)).astype(np.float32),
            rng.standard_normal((k, n)).astype(np.float32),
        ],
        lambda x, y: x @ y,
    )
    s = Schedule(w.func)
    out = s.cache_write("C", 0, "local")
    i, j, kk = s.get_loops("C")
    io, ii = s.split(i, [None, tile])
    jo, ji = s.split(j, [None, tile])
    s.reorder(io, jo, ii, ji, kk)
    s.reverse_compute_at(out, jo)
    init = s.decompose_reduction("C", ii)
    s.bind(io, "threadgroup.y")
    s.bind(jo, "threadgroup.x")
    for block in (init, "C", out):
        loops = s.get_loops(block)
        s.bind(loops[2], "thread.y")
        s.bind(loops[3], "thread.x")
    return s, w


def test_gpu_matmul_with_a_per_thread_accumulator() -> None:
    s, w = gpu_matmul(16, 12, 8, 4)
    check(s, w)
    text = tir.script(s.func)
    assert text.count("thread='thread.x'") == 3


def test_bind_after_fuse_and_imperfect_split() -> None:
    x = te.placeholder((10, 24), DType.float32, "x")
    y = te.compute((10, 24), lambda i, j: x[i, j] * 2.0, "y")
    rng = np.random.default_rng(4)
    w = Workload(te.create_prim_func([x, y]), [rng.standard_normal((10, 24)).astype(np.float32)],
                 lambda v: v * 2)  # fmt: skip
    s = Schedule(w.func)
    g, t = s.split(s.fuse(*s.get_loops("y")), [None, 32])
    s.bind(t, "thread.x")
    s.bind(g, "threadgroup.x")
    check(s, w)


@settings(max_examples=60, deadline=None)
@given(
    st.lists(st.integers(1, 9), min_size=1, max_size=3),
    st.integers(1, 40),
    st.sampled_from(["gpu", "parallel", "vectorize"]),
)
def test_concurrent_loops_after_fuse_and_split(shape: list[int], factor: int, how: str) -> None:
    x = te.placeholder(tuple(shape), DType.float32, "x")
    y = te.compute(tuple(shape), lambda *i: x[i] + 1.0, "y")
    data = np.random.default_rng(5).standard_normal(shape).astype(np.float32)
    w = Workload(te.create_prim_func([x, y]), [data], lambda v: v + 1)
    s = Schedule(w.func)
    loops = s.get_loops("y")
    fused = s.fuse(*loops) if len(loops) > 1 else loops[0]
    outer, inner = s.split(fused, [None, factor])
    if how == "gpu":
        s.bind(inner, "thread.x")
        s.bind(outer, "threadgroup.x")
    elif how == "parallel":
        s.parallel(outer)
        s.unroll(inner)
    else:
        s.parallel(outer)
        s.vectorize(inner)
    check(s, w)


def test_bind_refuses_same_axis_twice_on_a_path() -> None:
    s = Schedule(matmul_bias_relu(4, 4, 4).func)
    i, j = s.get_loops("relu")
    s.bind(i, "thread.x")
    refuses(s, "bound to thread.x", s.bind, j, "thread.x")


def test_annotate() -> None:
    s = Schedule(matmul_bias_relu(4, 4, 4).func)
    i = s.get_loops("matmul")[0]
    s.annotate(i, "pragma_unroll", 4)
    s.annotate("matmul", "role", "main")
    s.annotate("matmul", "role", "anchor")
    assert loops(s, "matmul")[0].annotations == (("pragma_unroll", 4),)
    assert realize(s, "matmul").block.annotations == (("role", "anchor"),)


# -- inlining ------------------------------------------------------------------------------


def test_compute_inline() -> None:
    w = elementwise_chain(5, 3)
    s = Schedule(w.func)
    s.compute_inline("double")
    s.compute_inline("transpose")
    assert s.blocks() == ["relu"]
    assert "alloc_buffer" not in tir.script(s.func)
    check(s, w)


def test_compute_inline_refusals() -> None:
    s = Schedule(matmul_bias_relu(4, 4, 4).func)
    refuses(s, "reduction", s.compute_inline, "matmul")
    refuses(s, "output", s.compute_inline, "relu")


def test_reverse_compute_inline() -> None:
    w = elementwise_chain(5, 3)
    s = Schedule(w.func)
    s.reverse_compute_inline("relu")
    assert s.blocks() == ["double", "transpose"]
    check(s, w)
    s2 = Schedule(matmul_bias_relu(4, 4, 4).func)
    refuses(s2, "not a complete spatial block", s2.reverse_compute_inline, "relu")


# -- compute_at --------------------------------------------------------------------------------


def test_compute_at_infers_halo_region() -> None:
    w = box_filter(7, 5, 2)
    s = Schedule(w.func)
    h = s.get_loops("box")[0]
    ho, hi = s.split(h, [None, 3])
    s.compute_at("pad", ho)
    pad_loops = loops(s, "pad")
    assert [f.loop_var for f in pad_loops[:1]] == [ho]
    assert [f.extent for f in pad_loops[1:]] == [IntImm(5), IntImm(7), IntImm(2)]
    assert "where(h_0 * 3 + ax0 < 9)" in tir.script(s.func)
    check(s, w)
    s.compute_at("pad", hi)
    assert [f.extent for f in loops(s, "pad")[2:]] == [IntImm(3), IntImm(7), IntImm(2)]
    check(s, w)


def test_compute_at_reduction_producer() -> None:
    w = row_softmax(4, 6)
    s = Schedule(w.func)
    i, _j = s.get_loops("softmax")
    s.compute_at("rowsum", i)
    s.compute_at("rowmax", i)
    assert s.blocks() == ["rowmax", "rowsum", "softmax"]
    check(s, w)


def test_compute_at_refusals() -> None:
    w = row_softmax(4, 6)
    s = Schedule(w.func)
    i_sum = s.get_loops("rowsum")[0]
    refuses(s, "not every consumer", s.compute_at, "rowmax", i_sum)
    i_soft = s.get_loops("softmax")[0]
    refuses(s, "output", s.compute_at, "softmax", s.get_loops("rowmax")[0])
    s.compute_at("rowsum", i_soft)
    refuses(s, "already under", s.compute_at, "rowsum", i_soft)


def test_reverse_compute_at_epilogue() -> None:
    w = matmul_bias_relu(8, 6, 5)
    s = Schedule(w.func)
    i, j, _k = s.get_loops("matmul")
    io, ii = s.split(i, [None, 4])
    jo, ji = s.split(j, [None, 4])
    s.reorder(io, jo, ii, ji)
    s.reverse_compute_at("relu", jo)
    relu_loops = loops(s, "relu")
    assert [f.loop_var for f in relu_loops[:2]] == [io, jo]
    assert [f.extent for f in relu_loops[2:]] == [IntImm(4), IntImm(4)]
    check(s, w)


def test_reverse_compute_at_refuses_inside_reduction() -> None:
    w = matmul_bias_relu(4, 4, 4)
    s = Schedule(w.func)
    k = s.get_loops("matmul")[2]
    refuses(s, "still being reduced", s.reverse_compute_at, "relu", k)


def test_reverse_compute_at_refuses_non_box_tiles() -> None:
    w = elementwise_chain(6, 4)
    s = Schedule(w.func)
    f = s.fuse(*s.get_loops("double"))
    fo, _ = s.split(f, [None, 5])
    refuses(s, "not a box", s.reverse_compute_at, "transpose", fo)


# -- caches and reductions ----------------------------------------------------------------------


def test_cache_read_at_tile() -> None:
    w = matmul_bias_relu(8, 8, 8)
    s = Schedule(w.func)
    a_local = s.cache_read("matmul", 0, "local")
    assert a_local == "A_local"
    check(s, w)
    _i, _j, k = s.get_loops("matmul")
    ko, _ = s.split(k, [None, 4])
    s.compute_at(a_local, ko)
    assert [f.extent for f in loops(s, a_local)[-1:]] == [IntImm(4)]
    check(s, w)


def test_cache_write_then_reverse_compute_at() -> None:
    w = matmul_bias_relu(8, 8, 4)
    s = Schedule(w.func)
    out = s.cache_write("matmul", 0, "local")
    assert out == "matmul_local"
    i, j, _k = s.get_loops("matmul")
    io, ii = s.split(i, [None, 4])
    jo, ji = s.split(j, [None, 4])
    s.reorder(io, jo, ii, ji)
    s.reverse_compute_at(out, jo)
    s.reverse_compute_at("relu", jo)
    check(s, w)


def test_cache_refusals() -> None:
    s = Schedule(matmul_bias_relu(4, 4, 4).func)
    refuses(s, "read regions", s.cache_read, "matmul", 5, "local")
    refuses(s, "scope", s.cache_read, "matmul", 0, "texture")


def test_decompose_reduction() -> None:
    w = matmul_bias_relu(6, 6, 6)
    s = Schedule(w.func)
    _i, _j, k = s.get_loops("matmul")
    init = s.decompose_reduction("matmul", k)
    assert init == "matmul_init"
    b = realize(s, "matmul").block
    assert isinstance(b, Block)
    assert b.init is None
    assert len(s.get_loops(init)) == 2
    check(s, w)
    s2 = Schedule(w.func)
    i2, _, k2 = s2.get_loops("matmul")
    s2.reorder(k2, i2)
    refuses(s2, "reduction loop", s2.decompose_reduction, "matmul", i2)


def test_decompose_reduction_with_imperfect_tiles() -> None:
    w = matmul_bias_relu(7, 5, 6)
    s = Schedule(w.func)
    i, _j, k = s.get_loops("matmul")
    io, ii = s.split(i, [None, 4])
    ko, _ki = s.split(k, [None, 4])
    s.reorder(io, ko, ii)
    s.decompose_reduction("matmul", ko)
    check(s, w)


def test_moving_a_decomposed_reduction_is_refused() -> None:
    w = row_softmax(4, 6)
    s = Schedule(w.func)
    s.decompose_reduction("rowmax", s.get_loops("rowmax")[0])
    refuses(s, "init was split off", s.compute_at, "rowmax", s.get_loops("softmax")[0])


def test_failed_primitive_leaves_schedule_unchanged() -> None:
    s = Schedule(matmul_bias_relu(4, 4, 4).func)
    before, trace = s.func, list(s.trace)
    with pytest.raises(ScheduleError):
        s.parallel(s.get_loops("matmul")[2])
    assert s.func is before
    assert s.trace == trace


def test_constant_indices_on_extent_one_dimensions() -> None:
    a = te.placeholder((1, 4), DType.float32, "A")
    b = te.compute((1, 4), lambda i, j: a[0, j] * 2.0, "B", ["i", "j"])
    c = te.compute((1, 4), lambda i, j: b[0, j] + 1.0, "C", ["i", "j"])
    data = np.arange(4, dtype=np.float32).reshape(1, 4)
    w = Workload(te.create_prim_func([a, c]), [data], lambda v: v * 2 + 1)
    for steps in (
        lambda s: s.compute_inline("B"),
        lambda s: s.reverse_compute_inline("C"),
        lambda s: s.compute_at("B", s.get_loops("C")[1]),
        lambda s: s.reverse_compute_at("C", s.get_loops("B")[1]),
    ):
        s = Schedule(w.func)
        steps(s)
        check(s, w)


def test_reverse_compute_at_after_an_imperfect_split() -> None:
    # A GEMV (one row) tiled by two rows: the second row of every tile is cut off.
    a = te.placeholder((1, 16), DType.float32, "A")
    b = te.placeholder((16, 10), DType.float32, "B")
    r = te.reduce_axis(16, "k")
    c = te.compute((1, 10), lambda i, j: te.sum(a[i, r] * b[r, j], axis=r), "logits", ["m", "n"])
    rng = np.random.default_rng(6)
    w = Workload(
        te.create_prim_func([a, b, c]),
        [
            rng.standard_normal((1, 16)).astype(np.float32),
            rng.standard_normal((16, 10)).astype(np.float32),
        ],
        lambda x, y: x @ y,
    )
    s = Schedule(w.func)
    out = s.cache_write("logits", 0, "local")
    m, n, k = s.get_loops("logits")
    mo, mi = s.split(m, [None, 2])
    no, ni = s.split(n, [None, 4])
    s.reorder(mo, no, mi, ni, k)
    s.reverse_compute_at(out, no)
    check(s, w)
