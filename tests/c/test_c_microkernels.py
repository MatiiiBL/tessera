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
from tessera.backend.c.microkernels import SGEMM_KERNELS, header_files, host_supports
from tessera.dtype import DType
from tessera.ir.graph import NDArray
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
                     int64_t transposed, int64_t constant) {{
  if (transposed) tsr_{isa}_sgemm_bt(m, n, k, a, lda, b, ldb, c, ldc, accumulate, constant);
  else tsr_{isa}_sgemm(m, n, k, a, lda, b, ldb, c, ldc, accumulate, constant);
}}
TSR_EXPORT void conv(int64_t m, int64_t n, int64_t p0, const float* x, const float* w,
                     int64_t ldw, float* y, int64_t ldy, const int64_t* g, int64_t accumulate) {{
  tsr_{isa}_conv2d(m, n, p0, x, w, ldw, y, ldy, g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7],
                   g[8], g[9], g[10], g[11], accumulate, 0);
}}
TSR_EXPORT void forget(void) {{ tsr_forget_panels(); }}
"""


@functools.cache
def _library(isa: CpuIsa) -> ctypes.CDLL:
    files = {"wrapper.c": WRAPPER.format(isa=isa), **header_files([f"tessera_{isa}.h"])}
    return ctypes.CDLL(str(cbuild.build(files, "wrapper.c", Target(isa=isa))))


@functools.cache
def _gemm(isa: CpuIsa) -> Callable[..., None]:
    lib = _library(isa)
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
        int(accumulate), int(transposed), 0,
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
        IntImm(0), IntImm(0),
    )  # fmt: skip
    func = PrimFunc("tile", (a, b, c), root(Evaluate(Call(kernel.name, args, DType.int32))))
    rng = np.random.default_rng(3)
    av = rng.standard_normal((50, 16)).astype(np.float32)
    bv = rng.standard_normal((16, 24)).astype(np.float32)
    cv = np.zeros((40, 30), np.float32)
    run_kernel(func, [av, bv, cv], Target(isa=isa))
    np.testing.assert_allclose(cv[:, :24], av[10:] @ bv, rtol=1e-4, atol=1e-4)
    assert not cv[:, 24:].any()


def _conv_reference(
    x: NDArray, w: NDArray, stride: tuple[int, int], pad: tuple[int, int], dilation: tuple[int, int]
) -> NDArray:
    """NHWC (one image) by HWIO, padding ``pad`` on every side, in float64."""
    (h, wd, _), (kh, kw, _, o) = x.shape, w.shape
    (sh, sw), (ph, pw), (dh, dw) = stride, pad, dilation
    oh = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (wd + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    padded = np.pad(x.astype(np.float64), ((ph, ph), (pw, pw), (0, 0)))
    out = np.zeros((oh, ow, o))
    for i in range(kh):
        for j in range(kw):
            rows = slice(i * dh, i * dh + sh * (oh - 1) + 1, sh)
            cols = slice(j * dw, j * dw + sw * (ow - 1) + 1, sw)
            out += padded[rows, cols, :] @ w[i, j].astype(np.float64)
    return out


CONV_CASES = [
    # h, w, c, kh, kw, stride, pad, dilation, o
    (8, 8, 16, 3, 3, (1, 1), (1, 1), (1, 1), 24),
    (9, 7, 3, 7, 7, (2, 2), (3, 3), (1, 1), 16),  # ResNet's first layer, scaled down
    (6, 6, 8, 1, 1, (1, 1), (0, 0), (1, 1), 40),
    (10, 10, 12, 3, 3, (2, 2), (1, 1), (1, 1), 20),
    (11, 9, 5, 3, 3, (1, 2), (2, 1), (2, 2), 13),
]


@pytest.mark.parametrize("isa", ISA_PARAMS)
@pytest.mark.parametrize("case", CONV_CASES)
def test_conv2d_tiles(isa: CpuIsa, case: tuple) -> None:  # type: ignore[type-arg]
    h, wd, c, kh, kw, stride, pad, dilation, o = case
    rng = np.random.default_rng(h * 31 + c)
    x = rng.standard_normal((h, wd, c)).astype(np.float32)
    w = rng.standard_normal((kh, kw, c, o)).astype(np.float32)
    ref = _conv_reference(x, w, stride, pad, dilation)
    oh, ow = ref.shape[:2]
    pixels = oh * ow
    # A tile in the middle: some pixels and a range of output channels.
    p0, m = pixels // 5, pixels - pixels // 5 - pixels // 7
    o0, n = o // 4, o - o // 4 - 1
    y = rng.standard_normal((pixels, o)).astype(np.float32)
    before = y.copy()
    geometry = np.array([h, wd, c, kh, kw, *stride, *pad, *dilation, ow], np.int64)
    fn = _library(isa).conv
    fn.argtypes = (
        [ctypes.c_int64] * 3
        + [ctypes.c_void_p] * 2
        + [ctypes.c_int64]
        + [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_int64,
        ]
    )
    for accumulate in (0, 1):
        y[...] = before
        fn(
            m, n, p0, x.ctypes.data, w.ctypes.data + o0 * 4, o,
            y.ctypes.data + (p0 * o + o0) * 4, o, geometry.ctypes.data, accumulate,
        )  # fmt: skip
        want = ref.reshape(pixels, o)[p0 : p0 + m, o0 : o0 + n]
        if accumulate:
            want = want + before[p0 : p0 + m, o0 : o0 + n]
        np.testing.assert_allclose(y[p0 : p0 + m, o0 : o0 + n], want, rtol=1e-4, atol=1e-4)
        outside = np.ones_like(y, bool)
        outside[p0 : p0 + m, o0 : o0 + n] = False
        np.testing.assert_array_equal(y[outside], before[outside])


@pytest.mark.parametrize("isa", ISA_PARAMS)
@pytest.mark.parametrize("transposed", [False, True])
def test_constant_b_is_packed_once_and_forgotten(isa: CpuIsa, transposed: bool) -> None:
    """A constant B is packed at the first call and reused; after forget, new
    data at the same address is packed again."""
    m, n, k = 40, 70, 300
    rng = np.random.default_rng(5)
    a = rng.standard_normal((m, k)).astype(np.float32)
    b = rng.standard_normal((n, k) if transposed else (k, n)).astype(np.float32)
    c = np.zeros((m, n), np.float32)
    gemm, lib = _gemm(isa), _library(isa)
    lib.forget()

    def call() -> NDArray:
        ldb = b.shape[1]
        gemm(m, n, k, a.ctypes.data, k, b.ctypes.data, ldb, c.ctypes.data, n, 0, int(transposed), 1)
        return c.copy()

    def want() -> NDArray:
        bm = b.T if transposed else b
        return (a.astype(np.float64) @ bm.astype(np.float64)).astype(np.float32)

    np.testing.assert_allclose(call(), want(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(call(), want(), rtol=1e-4, atol=1e-4)  # from the kept packing
    old = want()
    b[...] = rng.standard_normal(b.shape).astype(np.float32)
    np.testing.assert_allclose(call(), old, rtol=1e-4, atol=1e-4)  # still the kept packing
    lib.forget()
    np.testing.assert_allclose(call(), want(), rtol=1e-4, atol=1e-4)
