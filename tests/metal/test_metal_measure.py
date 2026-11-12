"""Timing kernels for the tuner."""

from __future__ import annotations

import numpy as np
import pytest
from metal_funcs import bound, matmul_bound, matmul_naive, prim_func, requires_gpu

from tessera.backend.base import MeasureError
from tessera.backend.metal import backend
from tessera.backend.metal.codegen import INTRINSICS, register_intrinsic
from tessera.target import Target
from tessera.tir.expr import Call, FloatImm, Var
from tessera.tir.stmt import Buffer, BufferStore, Evaluate

pytestmark = requires_gpu
METAL = Target.parse("metal")


def test_measure_returns_gpu_seconds_per_call() -> None:
    rng = np.random.default_rng(0)
    small = backend().measure(matmul_bound(64, 64, 64, rng).func, METAL, min_seconds=0.01)
    large = backend().measure(matmul_bound(256, 256, 256, rng).func, METAL, min_seconds=0.01)
    # 64 times the multiply-adds on the same schedule; small kernels also pay a
    # fixed cost per launch.
    assert 0 < 4 * small < large


def test_measure_is_repeatable() -> None:
    func = matmul_bound(128, 128, 128, np.random.default_rng(0)).func
    times = [backend().measure(func, METAL, repeat=3, min_seconds=0.01) for _ in range(3)]
    assert max(times) < 2 * min(times)


def test_generation_errors_become_measure_errors() -> None:
    out = Buffer("out", (4096,))
    t = Var("t")
    func = prim_func(
        "too_wide", [out], bound(t, 4096, "thread.x", BufferStore(out, FloatImm(1.0), (t,)))
    )
    with pytest.raises(MeasureError, match="too_wide: 4096 threads"):
        backend().measure(func, METAL)


def test_compiler_errors_become_measure_errors() -> None:
    register_intrinsic("metal.broken", lambda writer, call: "this is not MSL;")
    try:
        out = Buffer("out", (1,))
        func = prim_func("broken", [out], Evaluate(Call("metal.broken", ())))
        with pytest.raises(MeasureError, match="MSL compilation failed"):
            backend().measure(func, METAL)
    finally:
        del INTRINSICS["metal.broken"]


def test_measure_checks_the_target() -> None:
    with pytest.raises(ValueError, match="cannot build"):
        backend().measure(matmul_naive(2, 2, 2, np.random.default_rng(0)).func, Target.parse("c"))
