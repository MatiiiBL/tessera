"""Tensorize matches a loop nest against an intrinsic's description and replaces it."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tessera import te, tir
from tessera.dtype import DType
from tessera.tir import intrin
from tessera.tir.expr import Call, IntImm, Var
from tessera.tir.schedule import Schedule, ScheduleError
from tessera.tir.stmt import (
    Evaluate,
    Stmt,
)

from .workloads import matmul_bias_relu, run

MATCHES: list[intrin.IntrinMatch] = []


def gemm_impl(match: intrin.IntrinMatch) -> Stmt:
    MATCHES.append(match)
    lda, ldb, ldc = (s[0] for s in match.strides)
    args = (*match.pointers, IntImm(match.sizes["K"]), IntImm(lda), IntImm(ldb), IntImm(ldc))
    return Evaluate(Call("test.sgemm_4x4", args))


def gemm_reference(a: Any, b: Any, c: Any, m: int, n: int, k: int) -> None:
    c += a @ b


intrin.register("test.gemm_4x4", intrin.matmul_desc(4, 4), gemm_impl, gemm_reference, override=True)


def tiled(m: int = 8, n: int = 8, k: int = 6) -> tuple[Schedule, Var]:
    w = matmul_bias_relu(m, n, k)
    s = Schedule(w.func)
    i, j, kk = s.get_loops("matmul")
    io, ii = s.split(i, [None, 4])
    jo, ji = s.split(j, [None, 4])
    s.reorder(io, jo, ii, ji, kk)
    s.decompose_reduction("matmul", ii)
    return s, ii


def test_tensorize_replaces_the_nest_and_keeps_the_result() -> None:
    w = matmul_bias_relu(8, 8, 6)
    s, ii = tiled()
    MATCHES.clear()
    s.tensorize(ii, "test.gemm_4x4")
    tir.verify(s.func)
    (match,) = MATCHES
    assert [b.name for b in match.buffers] == ["A", "B", "matmul"]
    assert match.extents == (4, 4, 6)
    assert match.sizes == {"K": 6}
    assert match.strides == ((6, 1), (8, 1), (8, 1))
    assert [tir.script(p) for p in match.pointers] == [
        "&A[i_0 * 4, 0]", "&B[0, j_0 * 4]", "&matmul[i_0 * 4, j_0 * 4]",
    ]  # fmt: skip
    text = tir.script(s.func)
    assert "test.sgemm_4x4(&A[i_0 * 4, 0], &B[0, j_0 * 4], &matmul[i_0 * 4, j_0 * 4], 6" in text
    assert "'tensorize': 'test.gemm_4x4'" in text
    np.testing.assert_allclose(run(s.func, w), w.expected(), rtol=1e-5, atol=1e-5)


def test_tensorize_refusals() -> None:
    s, ii = tiled(8, 8, 6)
    before = s.func
    with pytest.raises(ScheduleError, match="unknown intrinsic"):
        s.tensorize(ii, "test.nothing")
    jo = s.get_loops("matmul")[1]
    with pytest.raises(ScheduleError, match="perfect loop nest"):
        s.tensorize(jo, "test.gemm_4x4")
    assert s.func is before
    w = matmul_bias_relu(8, 8, 6)
    s2 = Schedule(w.func)
    i, j, k = s2.get_loops("matmul")
    io, ii2 = s2.split(i, [None, 4])
    jo2, ji = s2.split(j, [None, 4])
    s2.reorder(io, jo2, ii2, ji, k)
    with pytest.raises(ScheduleError, match="decompose the reduction"):
        s2.tensorize(ii2, "test.gemm_4x4")
    s3 = Schedule(w.func)
    i, j, k = s3.get_loops("matmul")
    s3.decompose_reduction("matmul", i)
    with pytest.raises(ScheduleError, match="needs 4"):
        s3.tensorize(i, "test.gemm_4x4")


def test_tensorize_rejects_a_different_computation() -> None:
    a = te.placeholder((4, 3), DType.float32, "A")
    b = te.placeholder((3, 4), DType.float32, "B")
    k = te.reduce_axis(3, "k")
    c = te.compute((4, 4), lambda i, j: te.sum(a[i, k] + b[k, j], axis=k), "C")
    s = Schedule(te.create_prim_func([a, b, c]))
    i = s.get_loops("C")[0]
    s.decompose_reduction("C", i)
    with pytest.raises(ScheduleError, match="does not match"):
        s.tensorize(i, "test.gemm_4x4")
    t = te.compute((4, 4), lambda i, j: te.sum(a[j, k] * b[k, i], axis=k), "T")
    s = Schedule(te.create_prim_func([a, b, t]))
    i = s.get_loops("T")[0]
    s.decompose_reduction("T", i)
    with pytest.raises(ScheduleError, match=r"does not follow|does not match"):
        s.tensorize(i, "test.gemm_4x4")


def test_tensorized_functions_lower_and_keep_intrinsic_buffers() -> None:
    w = matmul_bias_relu(8, 8, 6)
    s, ii = tiled()
    a_local = s.cache_read("matmul", 0, "local")
    s.compute_at(a_local, s.get_loops("matmul")[1])
    ii = s.get_loops("matmul")[2]
    s.tensorize(ii, "test.gemm_4x4")
    tir.verify(s.func)
    np.testing.assert_allclose(run(s.func, w), w.expected(), rtol=1e-5, atol=1e-5)
    lowered = tir.lower_blocks(s.func)
    np.testing.assert_allclose(run(lowered, w), w.expected(), rtol=1e-5, atol=1e-5)
    compact = tir.compact_buffers(s.func)
    np.testing.assert_allclose(run(compact, w), w.expected(), rtol=1e-5, atol=1e-5)
    # A_local is passed by address with the strides of its full shape, so it stays whole.
    assert "A_local = alloc_buffer(float32[8, 6], 'local')" in tir.script(compact)


def test_matmul_desc_matches_a_lowered_matmul_group() -> None:
    from tessera.compute import lower_group
    from tessera.ir import Graph, TensorType
    from tessera.passes import fuse

    g = Graph("mm")
    x = g.add_input("x", TensorType((8, 5), DType.float32))
    y = g.add_input("y", TensorType((5, 12), DType.float32))
    g.set_outputs([g.op("MatMul", x, y)])
    (group,) = fuse(g)
    lowered = lower_group(group)
    assert lowered.main_block is not None
    s = Schedule(lowered.func)
    m, n, k = s.get_loops(lowered.main_block)
    mo, mi = s.split(m, [None, 4])
    no, ni = s.split(n, [None, 4])
    s.reorder(mo, no, mi, ni, k)
    s.decompose_reduction(lowered.main_block, mi)
    s.tensorize(mi, "test.gemm_4x4")
    rng = np.random.default_rng(1)
    a = rng.standard_normal((8, 5)).astype(np.float32)
    b = rng.standard_normal((5, 12)).astype(np.float32)
    out = np.zeros((8, 12), np.float32)
    tir.interpret(s.func, [a, b, out])
    np.testing.assert_allclose(out, a @ b, rtol=1e-5, atol=1e-5)


# -- edge tiles of imperfect splits --------------------------------------------------------


@pytest.mark.parametrize(("m", "n"), [(8, 10), (7, 10), (6, 37)])
def test_tensorize_edge_tiles_of_imperfect_splits(m: int, n: int) -> None:
    w = matmul_bias_relu(m, n, 5)
    s, ii = tiled(m, n, 5)
    MATCHES.clear()
    s.tensorize(ii, "test.gemm_4x4")
    tir.verify(s.func)
    (match,) = MATCHES
    assert match.extents == (4, 4, 5)
    valid = [tir.script(v) for v in match.valid]
    assert valid[0] == ("4" if m % 4 == 0 else f"min({m} - i_0 * 4, 4)")
    assert valid[1] == f"min({n} - j_0 * 4, 4)"
    assert valid[2] == "5"
    # The regions are cut to the valid part; the pointers stay at the tile's start.
    c_region = tir.script(match.regions[2].region[1].extent)
    assert c_region == valid[1]
    assert tir.script(match.pointers[2]) == "&matmul[i_0 * 4, j_0 * 4]"
    np.testing.assert_allclose(run(s.func, w), w.expected(), rtol=1e-5, atol=1e-5)
    lowered = tir.lower_blocks(s.func)
    np.testing.assert_allclose(run(lowered, w), w.expected(), rtol=1e-5, atol=1e-5)


def test_tensorize_the_edge_of_a_wide_output() -> None:
    # The LM-head shape in small: N = 50257 split by 64 leaves an edge tile of 17.
    a = te.placeholder((4, 3), DType.float32, "A")
    b = te.placeholder((3, 50257), DType.float32, "B")
    r = te.reduce_axis(3, "k")
    c = te.compute((4, 50257), lambda i, j: te.sum(a[i, r] * b[r, j], axis=r), "C", ["i", "j"])
    intrin.register("test.gemm_4x64", intrin.matmul_desc(4, 64), gemm_impl, gemm_reference,
                    override=True)  # fmt: skip
    s = Schedule(te.create_prim_func([a, b, c]))
    i, j, k = s.get_loops("C")
    jo, ji = s.split(j, [None, 64])
    s.reorder(jo, i, ji, k)
    s.decompose_reduction("C", i)
    MATCHES.clear()
    s.tensorize(i, "test.gemm_4x64")
    tir.verify(s.func)
    assert tir.script(MATCHES[0].valid[1]) == "min(50257 - j_0 * 64, 64)"
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 3)).astype(np.float32)
    y = rng.standard_normal((3, 50257)).astype(np.float32)
    out = np.full((4, 50257), np.nan, np.float32)
    tir.interpret(s.func, [x, y, out])
    np.testing.assert_allclose(out, x @ y, rtol=1e-5, atol=1e-5)


def test_tensorize_refuses_other_predicates() -> None:
    # An imperfect split of the reduction loop cuts the nest along k.
    w = matmul_bias_relu(8, 8, 6)
    s = Schedule(w.func)
    i, j, k = s.get_loops("matmul")
    io, ii = s.split(i, [None, 4])
    jo, ji = s.split(j, [None, 4])
    ko, ki = s.split(k, [None, 4])
    s.reorder(io, jo, ko, ii, ji, ki)
    s.decompose_reduction("matmul", ko)
    before = s.func
    with pytest.raises(ScheduleError, match="reduction loop"):
        s.tensorize(ii, "test.gemm_4x4")
    assert s.func is before
    # A condition relating two matched loops is not an edge-tile bound.
    from tessera.tir.expr import lt
    from tessera.tir.primitives.common import find_block, realize_of, rebuild, with_body
    from tessera.tir.stmt import BlockRealize

    s2, ii2 = tiled(8, 8, 6)
    path = find_block(s2.func, "matmul")
    r = realize_of(path)
    loops = s2.get_loops("matmul")
    triangle = BlockRealize(r.iter_values, lt(loops[3], loops[2] + 1), r.block)
    s3 = Schedule(with_body(s2.func, rebuild(path, triangle)))
    with pytest.raises(ScheduleError, match="not an edge-tile bound"):
        s3.tensorize(ii2, "test.gemm_4x4")
