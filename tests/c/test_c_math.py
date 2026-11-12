"""Accuracy of the runtime's vectorizable exp, log, tanh, erf and sigmoid."""

from __future__ import annotations

import ctypes
import functools
import math
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from tessera.backend.c import build as cbuild
from tessera.backend.c.microkernels import header_text
from tessera.ir.graph import NDArray
from tessera.target import Target

FUNCTIONS = ["expf", "logf", "tanhf", "erff", "sigmoidf"]

SOURCE = """
#define TSR_NUM_THREADS 1
#include "tessera_runtime.h"
{functions}
"""
LOOP = """
TSR_EXPORT void apply_{name}(const float* restrict x, float* restrict y, int64_t n) {{
  #pragma clang loop vectorize(enable)
  for (int64_t i = 0; i < n; ++i) y[i] = tsr_{name}(x[i]);
}}
"""


def _source() -> dict[str, str]:
    loops = "".join(LOOP.format(name=name) for name in FUNCTIONS)
    return {
        "math.c": SOURCE.format(functions=loops),
        "tessera_runtime.h": header_text("tessera_runtime.h"),
    }


@functools.cache
def _library() -> ctypes.CDLL:
    return ctypes.CDLL(str(cbuild.build(_source(), "math.c", Target())))


def _apply(name: str, x: NDArray) -> NDArray:
    fn = getattr(_library(), f"apply_{name}")
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    xs = np.ascontiguousarray(x, np.float32)
    ys = np.empty_like(xs)
    fn(xs.ctypes.data, ys.ctypes.data, xs.size)
    return ys


def _sigmoid(v: NDArray) -> NDArray:
    out: NDArray = 1 / (1 + np.exp(-v))
    return out


REFERENCES: dict[str, tuple[Callable[[NDArray], NDArray], NDArray, float]] = {
    # function: (float64 reference, inputs, max error in ulp)
    "expf": (np.exp, np.linspace(-104, 88.72, 1_000_001), 1.5),
    "logf": (np.log, np.geomspace(1e-44, 3e38, 1_000_001), 1.0),
    "tanhf": (np.tanh, np.linspace(-12, 12, 1_000_001), 5.0),
    "erff": (np.vectorize(math.erf), np.linspace(-6, 6, 200_001), 6.0),
    "sigmoidf": (_sigmoid, np.linspace(-90, 90, 1_000_001), 3.0),
}


@pytest.mark.parametrize("name", FUNCTIONS)
def test_error_in_ulp(name: str) -> None:
    reference, inputs, bound = REFERENCES[name]
    x = inputs.astype(np.float32)
    got = _apply(name, x).astype(np.float64)
    want = reference(x.astype(np.float64))
    normal = (np.abs(want) > 1.2e-38) & (np.abs(want) < 3.4e38)
    ulp = np.spacing(np.abs(want[normal]).astype(np.float32)).astype(np.float64)
    error = np.abs(got[normal] - want[normal]) / ulp
    assert error.max() <= bound, f"{name}: {error.max():.2f} ulp at x={x[normal][error.argmax()]}"
    if name in ("tanhf", "erff", "sigmoidf"):
        # Absolute error, which is what activations like GELU see.
        assert np.abs(got - want).max() < 4e-7
    # Below the normal range the error is at most one subnormal step.
    assert np.all(
        np.abs(got[~normal & np.isfinite(want)] - want[~normal & np.isfinite(want)]) < 1e-37
    )


def test_special_values() -> None:
    inf, nan = np.inf, np.nan
    np.testing.assert_array_equal(
        _apply("expf", np.array([inf, -inf, 0, 89, -200], np.float32)), [inf, 0, 1, inf, 0]
    )
    assert np.isnan(_apply("expf", np.array([nan], np.float32))[0])
    np.testing.assert_array_equal(_apply("logf", np.array([0, inf, 1], np.float32)), [-inf, inf, 0])
    assert np.isnan(_apply("logf", np.array([-1, nan], np.float32))).all()
    np.testing.assert_array_equal(_apply("tanhf", np.array([inf, -inf, 0], np.float32)), [1, -1, 0])
    np.testing.assert_array_equal(_apply("erff", np.array([inf, -inf, 0], np.float32)), [1, -1, 0])
    np.testing.assert_array_equal(_apply("sigmoidf", np.array([inf, -inf], np.float32)), [1, 0])
    for name in ("tanhf", "erff", "sigmoidf"):
        assert np.isnan(_apply(name, np.array([nan], np.float32))[0])


def test_loops_using_them_are_vectorized(tmp_path: Path) -> None:
    for name, text in _source().items():
        (tmp_path / name).write_text(text)
    target = Target()
    flags = [f for f in cbuild.compile_flags(target) if f not in ("-shared", "-fPIC")]
    subprocess.run([target.cc, *flags, "-S", "math.c", "-o", "math.s"], cwd=tmp_path, check=True)
    asm = (tmp_path / "math.s").read_text()
    assert not re.search(r"\b(bl|call[q]?)\s+_?(expf|logf|tanhf|erff)\b", asm)
    assert re.search(r"(fmla(\.4s|\s+v\d+\.4s)|vfmadd\w*ps|mulps)", asm)
