"""Reductions bound to threads: SIMD-group and threadgroup combination."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from metal_funcs import (
    assert_close,
    block,
    bound,
    gpu,
    loop,
    prim_func,
    reduce,
    row_reduce,
    run_case,
    spatial,
)

from tessera.backend.metal.codegen import CodegenError, generate
from tessera.tir.expr import Binary, FloatImm, Var
from tessera.tir.stmt import Buffer, BufferStore

RNG = np.random.default_rng(3)


def test_wide_reductions_use_simd_functions_and_threadgroup_memory() -> None:
    kernel = generate(row_reduce(4, 1000, RNG, threads=256).func)
    source = kernel.source
    assert "out_partial = 0.0f;" in source
    assert "out_partial = simd_sum(out_partial);" in source
    assert "threadgroup float out_red[8];" in source
    assert source.count("threadgroup_barrier(mem_flags::mem_threadgroup);") == 2
    assert "if (tx % 32 == 0) {" in source
    assert "out_red[tx / 32] = out_partial;" in source
    # One thread writes the row: the init value combined with the total.
    assert "if (tx == 0 && vi_1 < 4) {\n    out[vi_1] = 0.0f + out_partial;" in source


def test_the_write_has_its_own_block_variables() -> None:
    from tessera.backend.metal.runtime import available

    kernel = generate(row_reduce(3, 64, RNG, threads=64).func)
    assert kernel.source.count("const int vi = ") == 1
    if available():
        from tessera.backend.metal.codegen import module_source
        from tessera.backend.metal.runtime import device

        device().compile(module_source([kernel]))


def test_narrow_reductions_shuffle_within_the_simd_group() -> None:
    source = generate(row_reduce(64, 40, RNG, threads=8, op="max", rows_per_group=4).func).source
    assert "for (ushort offset = 4; offset > 0; offset >>= 1) {" in source
    assert "out_partial = max(out_partial, simd_shuffle_xor(out_partial, offset));" in source
    assert "out_partial = -INFINITY;" in source
    assert "threadgroup_barrier" not in source


def test_rows_on_thread_y_index_the_scratch() -> None:
    source = generate(row_reduce(8, 300, RNG, threads=64, op="min", rows_per_group=4).func).source
    assert "out_red[ty * 2 + tx / 32] = out_partial;" in source
    assert "out_partial = out_red[ty * 2];" in source


def _sum_block(op: str, tag: str) -> object:
    x, out = Buffer("x", (4, 64)), Buffer("out", (4,))
    r, t, vi, vk = Var("r"), Var("t"), Var("vi"), Var("vk")
    body = bound(
        r,
        4,
        "threadgroup.x",
        bound(
            t,
            64,
            tag,
            block(
                "out",
                [spatial(vi, 4, r), reduce(vk, 64, t)],
                BufferStore(out, Binary(op, out[vi], x[vi, vk]), (vi,)),  # type: ignore[arg-type]
                init=BufferStore(out, FloatImm(1.0), (vi,)),
            ),
        ),
    )
    return prim_func("bad", [x, out], body)


def test_unsupported_reductions_are_refused() -> None:
    with pytest.raises(CodegenError, match="sum, a maximum or a minimum"):
        generate(_sum_block("mul", "thread.x"))  # type: ignore[arg-type]
    with pytest.raises(CodegenError, match=r"run over thread\.x, not thread\.y"):
        generate(_sum_block("add", "thread.y"))  # type: ignore[arg-type]
    with pytest.raises(CodegenError, match="power of two up to 32 or a multiple of 32"):
        generate(row_reduce(4, 100, RNG, threads=48).func)


def test_reduction_loop_outside_a_spatial_loop_is_refused() -> None:
    x, out = Buffer("x", (4, 64)), Buffer("out", (4,))
    t, i, vi, vk = Var("t"), Var("i"), Var("vi"), Var("vk")
    body = bound(
        t,
        64,
        "thread.x",
        loop(
            i,
            4,
            block(
                "out",
                [spatial(vi, 4, i), reduce(vk, 64, t)],
                BufferStore(out, out[vi] + x[vi, vk], (vi,)),
            ),
        ),
    )
    with pytest.raises(CodegenError, match="innermost"):
        generate(prim_func("outside", [x, out], body))


@gpu
@given(
    threads=st.sampled_from([1, 2, 4, 8, 16, 32, 64, 128, 256, 1024]),
    k=st.integers(1, 700),
    op=st.sampled_from(["add", "max", "min"]),
    rows_per_group=st.sampled_from([1, 2, 3]),
    lanes_outer=st.booleans(),
    with_init=st.booleans(),
)
@settings(max_examples=40, deadline=None)
def test_thread_reductions_are_correct(
    threads: int, k: int, op: str, rows_per_group: int, lanes_outer: bool, with_init: bool
) -> None:
    if threads * rows_per_group > 1024:
        return
    case = row_reduce(
        5,
        k,
        np.random.default_rng(k),
        threads=threads,
        op=op,
        rows_per_group=rows_per_group,
        lanes_outer=lanes_outer,
        with_init=with_init,
    )
    result, reference = run_case(case)
    assert_close(result, reference, rtol=1e-5, atol=1e-4)
