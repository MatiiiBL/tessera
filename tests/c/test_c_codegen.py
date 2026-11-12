"""The C printer: structure and determinism of the text (nothing is compiled here)."""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np
import pytest
from c_funcs import R, S, loop, matmul, realize, root, tiled_matmul
from hypothesis import given
from hypothesis import strategies as st

from tessera.backend.c.codegen import CodegenError, float_literal, fold, generate_kernel
from tessera.backend.c.microkernels import Microkernel, register_microkernel
from tessera.dtype import DType
from tessera.tir.expr import Binary, Call, Expr, FloatImm, IntImm, Var
from tessera.tir.stmt import Buffer, BufferStore, Evaluate, For, ForKind, PrimFunc, seq


def test_kernel_is_static_with_restrict_pointers() -> None:
    text = generate_kernel(matmul(8, 16, 4)).text
    assert text.startswith(
        "static void matmul(float* restrict A, float* restrict B, float* restrict C) {"
    )
    assert "if (vk == 0) {" in text  # the init runs at the first reduce iteration
    assert "C[vi * 16 + vj] += A[vi * 4 + vk] * B[vk * 16 + vj];" in text


def test_generation_is_deterministic() -> None:
    # Two separately built, structurally equal functions print the same text.
    assert (
        generate_kernel(tiled_matmul(10, 32, 8)).text
        == generate_kernel(tiled_matmul(10, 32, 8)).text
    )


def test_parallel_loop_is_outlined_with_its_captures() -> None:
    code = generate_kernel(tiled_matmul(10, 32, 8))
    assert code.parallel
    assert "typedef struct {\n  float* A;\n  float* B;\n  float* C;\n} tiled_parallel0_args;" in (
        code.text
    )
    assert "tsr_parallel_for(0, 3, tiled_parallel0, &args);" in code.text
    body = code.text.split("static void tiled_parallel0_body")[1].split("static void")[0]
    # The full-shape local accumulator shrinks to one 4x16 tile per iteration.
    assert "float C_local[64]" in body
    assert "C_local[ii * 16 + ji] = 0.0f;" in body
    assert "#pragma clang loop vectorize(enable)" in body


def test_kernel_without_parallel_loops_does_not_use_the_pool() -> None:
    assert not generate_kernel(matmul(4, 4, 4)).parallel


def test_nested_parallel_loop_runs_serially() -> None:
    x, y = Buffer("X", (4, 8)), Buffer("Y", (4, 8))
    i, j, vi, vj = Var("i"), Var("j"), Var("vi"), Var("vj")
    block = realize("Y", [(vi, 4, S, i), (vj, 8, S, j)], BufferStore(y, x[vi, vj], (vi, vj)))
    body = loop(i, 4, loop(j, 8, block, ForKind.PARALLEL), ForKind.PARALLEL)
    text = generate_kernel(PrimFunc("nested", (x, y), root(body))).text
    assert text.count("tsr_parallel_for(") == 1
    assert "parallel loop inside a parallel loop: runs serially" in text


def test_names_are_sanitized_and_unique() -> None:
    x = Buffer("in/put.0", (4,))
    y = Buffer("for", (4,))
    i1, i2 = Var("i"), Var("i")
    v1, v2 = Var("1v"), Var("1v")
    first = realize("a", [(v1, 4, S, i1)], BufferStore(y, x[v1], (v1,)))
    second = realize("b", [(v2, 4, S, i2)], BufferStore(y, y[v2] * 2.0, (v2,)))
    func = PrimFunc("k", (x, y), root(seq([loop(i1, 4, first), loop(i2, 4, second)])))
    text = generate_kernel(func).text
    assert "static void k(float* restrict in_put_0, float* restrict for_1)" in text
    assert "for (int64_t i = 0; i < 4; ++i)" in text
    assert "for (int64_t i_1 = 0; i_1 < 4; ++i_1)" in text
    assert "const int64_t v1v = i;" in text
    assert "const int64_t v1v_1 = i_1;" in text


def _index_text(index_of: Callable[[Var], Expr]) -> str:
    """The kernel ``Y[i] = X[index_of(i)]`` for i in [0, 16)."""
    x, y = Buffer("X", (64,)), Buffer("Y", (64,))
    i = Var("i")
    body = loop(i, 16, BufferStore(y, x[index_of(i)], (i,)))
    return generate_kernel(PrimFunc("idx", (x, y), root(body))).text


def test_floordiv_by_a_power_of_two_is_a_shift() -> None:
    assert "X[i >> 2]" in _index_text(lambda i: i // 4)
    assert "X[i & 3]" in _index_text(lambda i: i % 4)


def test_floordiv_of_nonnegative_values_is_plain_division() -> None:
    assert "X[i / 3]" in _index_text(lambda i: i // 3)
    assert "X[i % 3]" in _index_text(lambda i: i % 3)


def test_floordiv_of_possibly_negative_values_uses_the_helper() -> None:
    assert "X[tsr_floordiv(i - 5, 3) + 2]" in _index_text(lambda i: (i - 5) // 3 + 2)
    assert "X[tsr_floormod(i - 5, 3)]" in _index_text(lambda i: (i - 5) % 3)


def test_thread_bound_loops_are_rejected() -> None:
    x = Buffer("X", (4,))
    i = Var("i")
    body = For(
        i,
        IntImm(0),
        IntImm(4),
        ForKind.THREAD_BINDING,
        BufferStore(x, FloatImm(1.0), (i,)),
        "thread.x",
    )
    with pytest.raises(CodegenError, match="Metal only"):
        generate_kernel(PrimFunc("bound", (x,), root(body)))


def _microkernel_func(name: str) -> PrimFunc:
    x = Buffer("X", (4, 8))
    return PrimFunc(
        "uses_mk",
        (x,),
        root(Evaluate(Call(name, (x.address(1, 0), IntImm(8)), DType.int32))),
    )


def test_microkernel_calls_need_a_registered_kernel_and_the_isa() -> None:
    register_microkernel(Microkernel("c.test_fill", "tsr_test_fill", "tessera_runtime.h", "neon"))
    with pytest.raises(CodegenError, match="no C microkernel"):
        generate_kernel(_microkernel_func("c.missing"))
    with pytest.raises(CodegenError, match="needs ISA 'neon'"):
        generate_kernel(_microkernel_func("c.test_fill"), "generic")
    code = generate_kernel(_microkernel_func("c.test_fill"), "sme")
    assert "tsr_test_fill(X + 8, 8);" in code.text
    assert code.headers == ("tessera_runtime.h",)


def test_reregistering_a_different_microkernel_fails() -> None:
    register_microkernel(Microkernel("c.test_once", "a", "tessera_runtime.h", "generic"))
    with pytest.raises(ValueError, match="already registered"):
        register_microkernel(Microkernel("c.test_once", "b", "tessera_runtime.h", "generic"))


def test_fold_drops_identities_and_folds_constants() -> None:
    i = Var("i")
    assert fold(i * 1 + 0) is i
    assert fold(Binary("mul", IntImm(3), IntImm(4)) + i * 0) == IntImm(12)
    assert fold(Binary("floordiv", IntImm(-7), IntImm(2))) == IntImm(-4)
    assert fold(Binary("floormod", IntImm(-7), IntImm(2))) == IntImm(1)
    assert fold(i + -3) == Binary("sub", i, IntImm(3))
    # Floats are left alone (x * 0 is not 0 for infinities).
    f = Var("f", DType.float32)
    assert fold(f * 0.0) == Binary("mul", f, FloatImm(0.0))


@given(st.floats(width=32, allow_nan=False, allow_infinity=False))
def test_float_literals_round_trip(value: float) -> None:
    text = float_literal(value)
    assert re.fullmatch(r"-?[0-9.]+(e[-+]?[0-9]+)?f", text), text
    assert np.float32(text[:-1]) == np.float32(value)


def test_special_float_literals() -> None:
    assert float_literal(float("inf")) == "INFINITY"
    assert float_literal(float("-inf")) == "-INFINITY"
    assert float_literal(float("nan")) == "NAN"
    assert float_literal(1.0, DType.float16) == "(_Float16)1.0f"


def test_reduction_block_with_reordered_loops_still_inits_once() -> None:
    a, c = Buffer("A", (4, 8)), Buffer("C", (4,))
    i, k, vi, vk = Var("i"), Var("k"), Var("vi"), Var("vk")
    block = realize(
        "C",
        [(vi, 4, S, i), (vk, 8, R, k)],
        BufferStore(c, c[vi] + a[vi, vk], (vi,)),
        init=BufferStore(c, FloatImm(0.0), (vi,)),
    )
    text = generate_kernel(PrimFunc("rowsum", (a, c), root(loop(k, 8, loop(i, 4, block))))).text
    assert text.index("for (int64_t k") < text.index("for (int64_t i")
    assert "if (vk == 0) {" in text
