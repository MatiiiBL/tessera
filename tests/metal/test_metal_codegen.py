"""MSL generation: launch shapes, barriers, names, index types. No GPU needed."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from metal_funcs import (
    add_relu,
    block,
    bound,
    bound_unary,
    exp_then_scale,
    loop,
    matmul_bound,
    matmul_naive,
    matmul_shared,
    prim_func,
    row_sum_tree,
    spatial,
)

from tessera.backend.metal.codegen import (
    BARRIER,
    CodegenError,
    fold,
    generate,
    insert_barriers,
    module_source,
)
from tessera.dtype import DType
from tessera.tir.expr import Call, Expr, FloatImm, IntImm, Var, call, lt
from tessera.tir.stmt import Buffer, BufferStore, Evaluate, ForKind, IfThenElse, Stmt, seq

RNG = np.random.default_rng(0)


def barrier_count(stmt: Stmt) -> int:
    from tessera.backend.metal.codegen import _walk_stmt

    return sum(
        1
        for s in _walk_stmt(stmt)
        if isinstance(s, Evaluate) and isinstance(s.value, Call) and s.value.name == BARRIER
    )


def test_unscheduled_function_is_one_thread() -> None:
    kernel = generate(add_relu(10, RNG).func)
    assert kernel.grid == (1, 1, 1)
    assert kernel.threads == (1, 1, 1)
    assert "for (int i = 0; i < 10; ++i)" in kernel.source
    assert "c[i] = max(a[i] + b[i], 0.0f);" in kernel.source
    assert [p.written for p in kernel.params] == [False, False, True]
    assert "device const float* a [[buffer(0)]]" in kernel.source
    assert "thread_pos" not in kernel.source


def test_thread_bindings_give_the_launch_shape() -> None:
    kernel = generate(matmul_bound(10, 12, 5, RNG, tile=8).func)
    assert kernel.grid == (2, 2, 1)
    assert kernel.threads == (8, 8, 1)
    assert "const int tx = (int)thread_pos.x;" in kernel.source
    assert "const int by = (int)threadgroup_pos.y;" in kernel.source
    assert "if (vi < 10 && vj < 12)" in kernel.source
    assert "[[max_total_threads_per_threadgroup(64)]]" in kernel.source


def test_reduction_init_runs_on_the_first_iteration() -> None:
    source = generate(matmul_naive(4, 5, 6, RNG).func).source
    assert "if (k == 0) {\n          C[i * 5 + j] = 0.0f;" in source
    assert "C[i * 5 + j] += A[i * 6 + k] * B[k * 5 + j];" in source


def test_shared_tiles_get_barriers_after_loading_and_before_reloading() -> None:
    kernel = generate(matmul_shared(64, RNG).func)
    assert kernel.threadgroup_memory == 2 * 16 * 16 * 4
    lines = [line.strip() for line in kernel.source.splitlines()]
    assert "threadgroup float A_shared[256];" in lines
    assert "float acc;" in lines
    loop_start = lines.index("for (int ko = 0; ko < 4; ++ko) {")
    body = lines[loop_start:]
    first = body.index("threadgroup_barrier(mem_flags::mem_threadgroup);")
    # The tiles are written before the first barrier and read after it.
    assert any(line.startswith("A_shared[") for line in body[:first])
    assert any("+= A_shared[" in line for line in body[first:])
    # The second barrier ends the iteration: the next one overwrites the tiles.
    second = body.index("threadgroup_barrier(mem_flags::mem_threadgroup);", first + 1)
    assert body[second + 1] == "}"
    assert "#pragma unroll" in lines


def test_tree_reduction_has_a_barrier_between_steps() -> None:
    func = row_sum_tree(4, RNG, threads=16).func
    # Five writes of `red` follow each other, each reading the previous step.
    assert barrier_count(insert_barriers(func).body) == 5


def test_barrier_pass_leaves_independent_statements_alone() -> None:
    sh = Buffer("sh", (8,), scope="shared")
    out = Buffer("out", (8,))
    a, b = Var("a"), Var("b")
    body = seq(
        [
            bound(a, 8, "thread.x", BufferStore(sh, FloatImm(1.0), (a,))),
            bound(b, 8, "thread.x", BufferStore(out, FloatImm(2.0), (b,))),
        ]
    )
    func = prim_func("independent", [out], body, alloc=[sh])
    assert barrier_count(insert_barriers(func).body) == 0


def test_loop_carried_barrier_only_when_the_body_conflicts_with_itself() -> None:
    sh = Buffer("sh", (8,), scope="shared")
    out = Buffer("out", (8,))
    k, t1, t2 = Var("k"), Var("t1"), Var("t2")
    body = loop(
        k,
        4,
        seq(
            [
                bound(t1, 8, "thread.x", BufferStore(sh, out[t1], (t1,))),
                bound(t2, 8, "thread.x", BufferStore(out, sh[7 - t2], (t2,))),
            ]
        ),
    )
    func = prim_func("carried", [out], body, alloc=[sh])
    new = insert_barriers(func).body
    assert barrier_count(new) == 2
    # With one iteration nothing is carried.
    single = prim_func("single", [out], loop(k, 1, body.body), alloc=[sh])
    assert barrier_count(insert_barriers(single).body) == 1


def test_smaller_thread_loop_is_guarded() -> None:
    sh = Buffer("sh", (100,), scope="shared")
    out = Buffer("out", (256,))
    t1, t2 = Var("t1"), Var("t2")
    body = seq(
        [
            bound(t1, 100, "thread.x", BufferStore(sh, FloatImm(3.0), (t1,))),
            bound(t2, 256, "thread.x", BufferStore(out, FloatImm(1.0), (t2,))),
        ]
    )
    kernel = generate(prim_func("guarded", [out], body, alloc=[sh]))
    assert kernel.threads == (256, 1, 1)
    assert "if (t1 < 100) {" in kernel.source


def test_barrier_under_a_thread_dependent_branch_is_rejected() -> None:
    sh = Buffer("sh", (8,), scope="shared")
    out = Buffer("out", (8,))
    t = Var("t")
    inner = seq([BufferStore(sh, FloatImm(1.0), (t,)), BufferStore(out, sh[7 - t], (t,))])
    body = bound(t, 8, "thread.x", IfThenElse(lt(t, 4), inner))
    with pytest.raises(CodegenError, match="barrier is needed inside a branch"):
        generate(prim_func("divergent", [out], body, alloc=[sh]))


def test_global_intermediate_becomes_scratch() -> None:
    kernel = generate(exp_then_scale(8, RNG).func)
    assert [b.name for b in kernel.scratch] == ["tmp"]
    assert "device float* tmp [[buffer(2)]]" in kernel.source


def test_global_intermediate_across_threadgroups_is_rejected() -> None:
    tmp = Buffer("tmp", (64,))
    out = Buffer("out", (64,))
    g, t, g2, t2 = Var("g"), Var("t"), Var("g2"), Var("t2")
    body = seq(
        [
            bound(
                g,
                4,
                "threadgroup.x",
                bound(t, 16, "thread.x", BufferStore(tmp, FloatImm(1.0), (g * 16 + t,))),
            ),
            bound(
                g2,
                4,
                "threadgroup.x",
                bound(
                    t2, 16, "thread.x", BufferStore(out, tmp[63 - g2 * 16 - t2], (g2 * 16 + t2,))
                ),
            ),
        ]
    )
    with pytest.raises(CodegenError, match="global memory"):
        generate(prim_func("scratch", [out], body, alloc=[tmp]))


def test_launch_limits_are_checked() -> None:
    out = Buffer("out", (2048,))
    t = Var("t")
    body = bound(t, 2048, "thread.x", BufferStore(out, FloatImm(1.0), (t,)))
    with pytest.raises(CodegenError, match="threads per threadgroup"):
        generate(prim_func("big", [out], body))
    sh = Buffer("sh", (16384,), scope="shared")
    with pytest.raises(CodegenError, match="threadgroup memory"):
        generate(prim_func("shared", [out], loop(t, 1, BufferStore(sh, out[0], (0,))), alloc=[sh]))
    with pytest.raises(CodegenError, match="unknown axis"):
        generate(
            prim_func(
                "tag",
                [out],
                loop(t, 4, BufferStore(out, out[0], (t,)), ForKind.THREAD_BINDING, "warp.x"),
            )
        )


def test_index_type_widens_only_when_needed() -> None:
    assert generate(add_relu(10, RNG).func).index_type == "int"
    huge = Buffer("huge", (2**31,), DType.int8)
    i = Var("i")
    func = prim_func("wide", [huge], loop(i, 2**31, BufferStore(huge, IntImm(0, DType.int8), (i,))))
    kernel = generate(func)
    assert kernel.index_type == "long"
    assert "for (long i = 0L; i < 2147483648L; ++i)" in kernel.source


def test_names_are_sanitized_and_unique() -> None:
    a = Buffer("conv1/weight:0", (4,))
    b = Buffer("exp", (4,))
    c = Buffer("exp", (4,))
    i, i2 = Var("i"), Var("i")
    body = seq(
        [
            loop(i, 4, BufferStore(b, call("exp", a[i]), (i,))),
            loop(i2, 4, BufferStore(c, b[i2], (i2,))),
        ]
    )
    kernel = generate(prim_func("3x/odd name", [a, b, c], body))
    assert kernel.name == "v_3x_odd_name"
    assert [p.name for p in kernel.params] == ["conv1_weight_0", "exp_", "exp__1"]
    assert "exp_[i] = exp(conv1_weight_0[i]);" in kernel.source
    assert "for (int i_1 = 0; i_1 < 4; ++i_1)" in kernel.source


def test_floor_division_uses_c_operators_only_when_operands_are_nonnegative() -> None:
    x = Buffer("x", (16,), DType.int32)
    out = Buffer("out", (16,), DType.int32)
    i = Var("i")
    nonneg = loop(i, 16, BufferStore(out, x[i // 4], (i % 4,)))
    source = generate(prim_func("nonneg", [x, out], nonneg)).source
    assert "x[i / 4]" in source
    assert "out[i % 4]" in source
    assert "tessera_floordiv" not in source
    j = Var("j")
    signed = loop(j, 16, BufferStore(out, (x[j] // 3) + (x[j] % 3), (j,)))
    kernel = generate(prim_func("signed", [x, out], signed))
    assert "tessera_floordiv(x[j], 3) + tessera_floormod(x[j], 3)" in kernel.source
    assert kernel.helpers == {"floordiv", "floormod"}
    assert "inline int tessera_floordiv" in module_source([kernel])


def test_math_intrinsics_print_metal_functions() -> None:
    def source_of(fn: Callable[[Expr], Expr]) -> str:
        return generate(bound_unary(64, RNG, fn, np.abs, name="f").func).source

    assert "tanh(clamp(x[vi], -9.0f, 9.0f))" in source_of(lambda v: call("tanh", v))
    assert "tessera_erf(x[vi])" in source_of(lambda v: call("erf", v))
    assert "1.0f / (1.0f + exp(-x[vi]))" in source_of(lambda v: call("sigmoid", v))
    assert "rsqrt(x[vi])" in source_of(lambda v: call("rsqrt", v))
    with pytest.raises(CodegenError, match="unknown intrinsic"):
        source_of(lambda v: call("cuda.shfl", v))


def test_float_literals_round_trip_float32() -> None:
    from tessera.backend.metal.codegen import _float_literal

    for value in (0.1, 1.0, -2.5, 1e-30, 3.4028235e38, 1 / 3):
        text, _ = _float_literal(value, DType.float32)
        assert text.endswith("f")
        assert np.float32(float(text[:-1])) == np.float32(value)
    assert _float_literal(float("-inf"), DType.float32)[0] == "-INFINITY"


def test_fold_simplifies_integer_identities() -> None:
    i = Var("i")
    assert fold(i * 1 + 0) is i
    assert fold((i * 0) + IntImm(3) * 4) == IntImm(12)
    assert fold(i // 1) is i
    assert fold((i + 3) // 8 * 8 + (i + 3) % 8) == i + 3


def test_generation_is_deterministic() -> None:
    for make in (
        lambda: matmul_shared(32, np.random.default_rng(1)),
        lambda: add_relu(5, np.random.default_rng(1)),
    ):
        assert generate(make().func).source == generate(make().func).source


def test_module_rejects_duplicate_kernel_names() -> None:
    kernel = generate(add_relu(4, RNG).func)
    with pytest.raises(CodegenError, match="not unique"):
        module_source([kernel, kernel])


def test_block_iteration_values_are_named_once() -> None:
    out = Buffer("out", (8, 8))
    t, vi, vj = Var("t"), Var("vi"), Var("vj")
    body = bound(
        t,
        64,
        "thread.x",
        block(
            "out",
            [spatial(vi, 8, t // 8), spatial(vj, 8, t % 8)],
            BufferStore(out, FloatImm(1.0), (vi, vj)),
        ),
    )
    source = generate(prim_func("named", [out], body)).source
    assert "const int vi = t / 8;" in source
    assert "out[vi * 8 + vj] = 1.0f;" in source


def test_negating_a_negative_literal_is_not_a_decrement() -> None:
    case = bound_unary(8, RNG, lambda v: call("sigmoid", FloatImm(-0.5)), np.abs)
    source = generate(case.func).source
    assert "exp(-(-0.5f))" in source


def test_thread_loops_with_an_offset() -> None:
    out = Buffer("out", (16,))
    t, u = Var("t"), Var("u")
    body = seq(
        [
            loop(
                t, 8, BufferStore(out, FloatImm(1.0), (t,)), ForKind.THREAD_BINDING, "thread.x", 8
            ),
            bound(u, 8, "thread.x", BufferStore(out, FloatImm(2.0), (u,))),
        ]
    )
    source = generate(prim_func("offset", [out], body)).source
    assert "const int t = (int)thread_pos.x;" in source
    assert "const int t_1 = 8 + t;" in source
    assert "out[t_1] = 1.0f;" in source
    assert "out[t] = 2.0f;" in source


def test_condition_on_a_substituted_thread_variable_is_divergent() -> None:
    sh = Buffer("sh", (8,), scope="shared")
    out = Buffer("out", (8,))
    t, u = Var("t"), Var("u")
    inner = seq([BufferStore(sh, FloatImm(1.0), (u,)), BufferStore(out, sh[7 - u], (u,))])
    body = seq(
        [
            bound(t, 8, "thread.x", BufferStore(out, FloatImm(0.0), (t,))),
            bound(u, 8, "thread.x", IfThenElse(lt(u, 4), inner)),
        ]
    )
    with pytest.raises(CodegenError, match="barrier is needed inside a branch"):
        generate(prim_func("divergent2", [out], body, alloc=[sh]))
