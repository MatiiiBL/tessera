"""SIMD-group matrices: fragment buffers, 32-lane SIMD groups, multiply-accumulate."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from metal_funcs import assert_close, bound, gpu, matmul_simdgroup, prim_func, run_case

from tessera.backend.metal.codegen import (
    CodegenError,
    generate,
    simdgroup_fill,
    simdgroup_load,
    simdgroup_store,
)
from tessera.tir.expr import FloatImm, IntImm, Var
from tessera.tir.stmt import Buffer, BufferStore, seq

RNG = np.random.default_rng(4)


def test_fragments_become_simdgroup_matrices_sized_for_one_simd_group() -> None:
    kernel = generate(matmul_simdgroup(64, 64, 32, RNG, groups=(1, 2), frags=(2, 4)).func)
    source = kernel.source
    # C_frag, A_frag and B_frag have the whole tensors' shapes in the IR; a SIMD
    # group holds 2 x 4 accumulator tiles and one column (row) of A (B) tiles.
    assert "simdgroup_float8x8 C_frag[8];" in source
    assert "simdgroup_float8x8 A_frag[2];" in source
    assert "simdgroup_float8x8 B_frag[4];" in source
    assert "C_frag[j + i * 4] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);" in source
    assert (
        "simdgroup_multiply_accumulate(C_frag[j_1 + i_3 * 4], A_frag[i_3], B_frag[j_1], " in source
    )
    assert (
        "simdgroup_load(A_frag[i_1], "
        "&A[(group_y * 16 + thread_z * 16 + i_1 * 8) * 32 + ko * 8], 32);" in source
    )
    # Two SIMD groups of 32 lanes, launched as one 64-thread row.
    assert kernel.shape == (32, 2, 1)
    assert kernel.threads == (64, 1, 1)
    assert "const int thread_y = (int)(thread_index / 32);" in source


def test_fragments_are_only_used_through_simdgroup_calls() -> None:
    a, out = Buffer("A", (8, 8)), Buffer("out", (8, 8))
    frag = Buffer("frag", (8, 8), scope="local")
    body = seq(
        [
            simdgroup_load(frag.address(0, 0), a.address(0, 0), 8),
            BufferStore(out, frag[0, 0], (IntImm(0), IntImm(0))),
        ]
    )
    with pytest.raises(CodegenError, match="only simdgroup intrinsics"):
        generate(prim_func("misuse", [a, out], body, alloc=[frag]))


def test_fragment_buffers_must_be_whole_tiles() -> None:
    out = Buffer("out", (8, 12))
    frag = Buffer("frag", (8, 12), scope="local")
    # Tiles at columns 0 and 4 overlap: together they span 12 columns, not whole tiles.
    body = seq(
        [
            simdgroup_fill(frag.address(0, 0), FloatImm(0.0)),
            simdgroup_fill(frag.address(0, 4), FloatImm(0.0)),
        ]
    )
    with pytest.raises(CodegenError, match="multiples of 8"):
        generate(prim_func("narrow", [out], body, alloc=[frag]))


def test_simdgroup_kernels_need_thread_x_for_lanes() -> None:
    out = Buffer("out", (8, 8))
    frag = Buffer("frag", (8, 8), scope="local")
    t = Var("t")
    body = bound(
        t,
        64,
        "thread.x",
        seq(
            [
                simdgroup_fill(frag.address(0, 0), FloatImm(0.0)),
                simdgroup_store(frag.address(0, 0), out.address(0, 0), 8),
            ]
        ),
    )
    with pytest.raises(CodegenError, match=r"thread\.x must be its 32 lanes"):
        generate(prim_func("lanes", [out], body, alloc=[frag]))


@gpu
def test_transposed_load() -> None:
    a, out = Buffer("A", (8, 8)), Buffer("out", (8, 8))
    frag = Buffer("frag", (8, 8), scope="local")
    body = seq(
        [
            simdgroup_load(frag.address(0, 0), a.address(0, 0), 8, transpose=True),
            simdgroup_store(frag.address(0, 0), out.address(0, 0), 8),
        ]
    )
    from tessera.backend.metal.executor import run_func

    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    result = np.zeros((8, 8), np.float32)
    run_func(prim_func("transpose", [a, out], body, alloc=[frag]), [data, result])
    np.testing.assert_array_equal(result, data.T)


@gpu
@given(
    groups=st.sampled_from([(1, 1), (1, 2), (2, 1), (2, 2), (1, 4)]),
    frags=st.sampled_from([(1, 1), (1, 2), (2, 2), (2, 4), (4, 1)]),
    tiles=st.tuples(st.integers(1, 3), st.integers(1, 3)),
    k=st.sampled_from([8, 16, 64, 136]),
)
@settings(max_examples=25, deadline=None)
def test_simdgroup_matmul_is_correct(
    groups: tuple[int, int], frags: tuple[int, int], tiles: tuple[int, int], k: int
) -> None:
    m = groups[0] * frags[0] * 8 * tiles[0]
    n = groups[1] * frags[1] * 8 * tiles[1]
    case = matmul_simdgroup(m, n, k, np.random.default_rng(k), groups=groups, frags=frags)
    result, reference = run_case(case)
    assert_close(result, reference, rtol=1e-4, atol=1e-4)


def test_the_interpreter_runs_simdgroup_calls() -> None:
    import tessera.tir as tir

    case = matmul_simdgroup(32, 64, 24, np.random.default_rng(5), groups=(2, 1), frags=(2, 4))
    result = np.zeros((32, 64), np.float32)
    tir.interpret(case.func, [*case.inputs, result])
    assert_close(result, case.inputs[0].astype(np.float64) @ case.inputs[1], rtol=1e-5, atol=1e-5)


@gpu
def test_shaped_threadgroups_compute_the_same() -> None:
    from tessera.backend.metal import runtime
    from tessera.backend.metal.codegen import module_source

    case = matmul_simdgroup(32, 64, 16, np.random.default_rng(6), groups=(2, 2), frags=(1, 2))
    kernel = generate(case.func, flat=False)
    assert kernel.threads == (32, 2, 2)
    assert "uint3 thread_pos [[thread_position_in_threadgroup]]" in kernel.source
    device = runtime.device()
    pipeline = device.compile(module_source([kernel])).pipeline(kernel.name)
    buffers = [device.buffer(4 * x.size) for x in (*case.inputs, np.zeros((32, 64)))]
    for buffer, x in zip(buffers, case.inputs, strict=False):
        buffer.view(np.float32, x.shape)[...] = x
    program = device.program()
    program.add(pipeline, kernel.grid, kernel.threads, [(b, 0) for b in buffers])
    program.run()
    result = buffers[2].view(np.float32, (32, 64))
    assert_close(result, case.inputs[0].astype(np.float64) @ case.inputs[1], rtol=1e-4, atol=1e-4)


@gpu
def test_padding_after_buffers_never_reaches_results() -> None:
    from metal_workloads import matmul, space

    from tessera.backend.metal import runtime
    from tessera.backend.metal.bench import schedule_with
    from tessera.backend.metal.codegen import module_source
    from tessera.backend.metal.templates import TEMPLATES

    # 100 columns: the last SIMD-group tile is partial, and the kernel reads whole
    # tiles of B, past the end of its last row, into the allocation's padding.
    workload = matmul(32, 100, 16, epilogue=False)
    values = space(TEMPLATES["matmul"], workload).default()
    assert values["strategy"] == "simdgroup"
    kernel = generate(schedule_with(workload.lowered, values))
    device = runtime.device()
    buffers = [device.buffer(p.nbytes) for p in kernel.buffers]
    for buffer, x in zip(buffers, workload.inputs, strict=False):
        buffer.view(np.float32, x.shape)[...] = x
    for buffer in buffers:
        assert buffer.allocated == buffer.nbytes + runtime.PADDING
        buffer.view(np.float32, (runtime.PADDING // 4,), buffer.nbytes, padding=True)[:] = np.nan
    pipeline = device.compile(module_source([kernel])).pipeline(kernel.name)
    program = device.program()
    program.add(pipeline, kernel.grid, kernel.threads, [(b, 0) for b in buffers])
    program.run()
    result = buffers[len(workload.inputs)].view(np.float32, workload.output.shape)
    assert np.isfinite(result).all()
    assert_close(result, workload.output, rtol=1e-4, atol=1e-4)
