"""Timing single kernels in a separate process."""

from __future__ import annotations

import pytest
from c_funcs import loop, matmul, root, tiled_matmul

from tessera.backend.base import MeasureError, get_backend
from tessera.backend.c import CBackend
from tessera.backend.c.measure import build_harness, harness_sources, run_harness
from tessera.target import Target
from tessera.tir.expr import FloatImm, IntImm, Var
from tessera.tir.stmt import Buffer, BufferStore, PrimFunc


def test_measure_returns_the_median_time() -> None:
    backend = get_backend("c")
    seconds = backend.measure(tiled_matmul(64, 64, 64), Target(num_threads=2), min_seconds=0.01)
    # 64^3 multiply-adds take microseconds, not seconds.
    assert 1e-8 < seconds < 1e-2


def test_harness_reports_one_sample_per_repeat() -> None:
    path = build_harness(matmul(16, 16, 16), Target())
    samples = run_harness(path, repeat=4, min_seconds=0.004)
    assert len(samples) == 4
    assert all(s > 0 for s in samples)


def test_prebuild_fills_the_cache_used_by_measure() -> None:
    func = matmul(12, 12, 12, name="prebuilt")
    backend = CBackend()
    backend.prebuild(func, Target())
    assert build_harness(func, Target()).exists()
    assert "random_unit" in harness_sources(func, Target())["harness.c"]


def test_build_errors_are_measure_errors() -> None:
    with pytest.raises(MeasureError, match="failed"):
        get_backend("c").measure(matmul(4, 4, 4), Target(cflags=("-fno-such-flag",)))


def test_crashes_are_measure_errors() -> None:
    # A store far outside the buffer: the harness process dies, not the caller.
    y = Buffer("Y", (4,))
    body = BufferStore(y, FloatImm(1.0), (IntImm(1 << 40),))
    with pytest.raises(MeasureError, match="crashed with SIG"):
        get_backend("c").measure(PrimFunc("crash", (y,), root(body)), Target())


def test_timeouts_are_measure_errors() -> None:
    x, y = Buffer("X", (1024,)), Buffer("Y", (1,))
    i, j = Var("i"), Var("j")
    store = BufferStore(y, y[0] * x[j % 1024] + x[i % 1024], (IntImm(0),))
    body = loop(i, 1 << 20, loop(j, 1 << 20, store))  # 10^12 dependent steps
    func = PrimFunc("forever", (x, y), root(body))
    with pytest.raises(MeasureError, match="timed out"):
        CBackend().measure(func, Target(), timeout=0.5)
