"""The NEON and SME GEMM microkernels, on every ISA this machine runs."""

from __future__ import annotations

import ctypes
import functools
from collections.abc import Callable

import numpy as np
import pytest
from c_funcs import root
from hypothesis import given, settings
from hypothesis import strategies as st

from tessera.backend.c import build as cbuild
from tessera.backend.c import run_kernel
from tessera.backend.c.microkernels import SGEMM_KERNELS, header_text, host_supports
from tessera.dtype import DType
from tessera.target import CpuIsa, Target
from tessera.tir.expr import Call, IntImm
from tessera.tir.stmt import Buffer, Evaluate, PrimFunc

ISAS: list[CpuIsa] = ["neon", "sme"]

pytestmark = pytest.mark.skipif(not host_supports("neon"), reason="needs an Arm CPU")


def _isa_param(isa: CpuIsa) -> object:
    marks = [pytest.mark.sme] if isa == "sme" else []
    skip = pytest.mark.skipif(not host_supports(isa), reason=f"this CPU has no {isa}")
    return pytest.param(isa, marks=[*marks, skip])


ISA_PARAMS = [_isa_param(isa) for isa in ISAS]

WRAPPER = """
#define TSR_NUM_THREADS 1
#include "tessera_runtime.h"
#include "tessera_{isa}.h"
TSR_EXPORT void gemm(int64_t m, int64_t n, int64_t k, const float* a, int64_t lda,
                     const float* b, int64_t ldb, float* c, int64_t ldc, int64_t accumulate,
                     int64_t transposed) {{
  if (transposed) tsr_{isa}_sgemm_bt(m, n, k, a, lda, b, ldb, c, ldc, accumulate);
  else tsr_{isa}_sgemm(m, n, k, a, lda, b, ldb, c, ldc, accumulate);
}}
"""


@functools.cache
def _gemm(isa: CpuIsa) -> Callable[..., None]:
    files = {
        "gemm.c": WRAPPER.format(isa=isa),
        "tessera_runtime.h": header_text("tessera_runtime.h"),
        f"tessera_{isa}.h": header_text(f"tessera_{isa}.h"),
    }
    lib = ctypes.CDLL(str(cbuild.build(files, "gemm.c", Target(isa=isa))))
    fn = lib.gemm
    fn.argtypes = (
        [ctypes.c_int64] * 3 + [ctypes.c_void_p, ctypes.c_int64] * 3 + [ctypes.c_int64] * 2
    )
    fn.restype = None
    return fn


def _check(isa: CpuIsa, m: int, n: int, k: int, *, transposed: bool, accumulate: bool) -> None:
    rng = np.random.default_rng(m * 10007 + n * 101 + k)
    # Leading dimensions larger than the tile: the kernels must honour them.
    a_full = rng.standard_normal((m, k + 3)).astype(np.float32)
    b_full = rng.standard_normal((n, k + 5) if transposed else (k, n + 5)).astype(np.float32)
    c_full = rng.standard_normal((m, n + 7)).astype(np.float32)
    a = a_full[:, :k]
    b = b_full[:, :k].T if transposed else b_full[:, :n]
    expected = a.astype(np.float64) @ b.astype(np.float64)
    if accumulate:
        expected += c_full[:, :n]
    untouched = c_full[:, n:].copy()
    _gemm(isa)(
        m, n, k,
        a_full.ctypes.data, k + 3,
        b_full.ctypes.data, b_full.shape[1],
        c_full.ctypes.data, n + 7,
        int(accumulate), int(transposed),
    )  # fmt: skip
    np.testing.assert_allclose(c_full[:, :n], expected, rtol=1e-4, atol=1e-4 * np.sqrt(k + 1))
    np.testing.assert_array_equal(c_full[:, n:], untouched)


@pytest.mark.parametrize("isa", ISA_PARAMS)
@pytest.mark.parametrize(
    ("m", "n", "k"),
    [(1, 1, 1), (8, 12, 4), (32, 32, 32), (7, 13, 5), (33, 50, 300), (100, 37, 1100), (64, 64, 0)],
)
@pytest.mark.parametrize("transposed", [False, True])
def test_sgemm_tiles(isa: CpuIsa, m: int, n: int, k: int, transposed: bool) -> None:
    _check(isa, m, n, k, transposed=transposed, accumulate=False)
    _check(isa, m, n, k, transposed=transposed, accumulate=True)


@pytest.mark.parametrize("isa", ISA_PARAMS)
@settings(max_examples=40, deadline=None)
@given(
    m=st.integers(1, 80),
    n=st.integers(1, 80),
    k=st.integers(1, 80),
    transposed=st.booleans(),
    accumulate=st.booleans(),
)
def test_sgemm_any_shape(
    isa: CpuIsa, m: int, n: int, k: int, transposed: bool, accumulate: bool
) -> None:
    _check(isa, m, n, k, transposed=transposed, accumulate=accumulate)


@pytest.mark.parametrize("isa", ISA_PARAMS)
def test_generated_code_calls_the_microkernel(isa: CpuIsa) -> None:
    """A hand-tensorized PrimFunc: C[0:40, 0:24] of a 40 x 30 output, from a
    pointer into the middle of A."""
    a, b, c = Buffer("A", (50, 16)), Buffer("B", (16, 24)), Buffer("C", (40, 30))
    kernel = SGEMM_KERNELS[isa][False]
    args = (
        IntImm(40), IntImm(24), IntImm(16),
        a.address(10, 0), IntImm(16),
        b.address(0, 0), IntImm(24),
        c.address(0, 0), IntImm(30),
        IntImm(0),
    )  # fmt: skip
    func = PrimFunc("tile", (a, b, c), root(Evaluate(Call(kernel.name, args, DType.int32))))
    rng = np.random.default_rng(3)
    av = rng.standard_normal((50, 16)).astype(np.float32)
    bv = rng.standard_normal((16, 24)).astype(np.float32)
    cv = np.zeros((40, 30), np.float32)
    run_kernel(func, [av, bv, cv], Target(isa=isa))
    np.testing.assert_allclose(cv[:, :24], av[10:] @ bv, rtol=1e-4, atol=1e-4)
    assert not cv[:, 24:].any()
