"""Timing single kernels in a separate process."""

from __future__ import annotations

import statistics
from collections.abc import Mapping

import numpy as np
import pytest
from c_funcs import loop, matmul, root, tiled_matmul

from tessera.backend.base import MeasureError, get_backend
from tessera.backend.c import CBackend
from tessera.backend.c.measure import (
    COLD_BYTES,
    MAX_COPIES,
    build_harness,
    constant_copies,
    harness_sources,
    run_harness,
)
from tessera.backend.c.microkernels import host_supports
from tessera.driver import Compilation, lower
from tessera.dtype import DType
from tessera.ir import Graph, TensorType
from tessera.target import Target
from tessera.tir.expr import FloatImm, IntImm, Var
from tessera.tir.stmt import Buffer, BufferStore, PrimFunc
from tessera.tune.log import BestConfigs, Record
from tessera.tune.space import KnobValue


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


def test_constant_parameters_rotate_through_cold_copies() -> None:
    func = matmul(64, 64, 64, name="weights")
    copies = constant_copies(func, frozenset({1}))
    weight = 64 * 64 * 4
    assert copies == MAX_COPIES or copies * weight >= COLD_BYTES
    text = harness_sources(func, Target(), frozenset({1}))["harness.c"]
    assert f"COPIES = {copies}" in text
    assert "p1[c]" in text
    assert "p0[c]" not in text
    assert constant_copies(func, frozenset()) == 1


def test_large_constants_need_one_copy() -> None:
    func = matmul(8, 4096, 4096, name="big")  # a 64 MiB weight
    assert constant_copies(func, frozenset({1})) == 1


def test_constant_indices_must_be_parameters() -> None:
    with pytest.raises(ValueError, match="out of range"):
        harness_sources(matmul(4, 4, 4), Target(), frozenset({3}))


def test_measure_with_constants() -> None:
    func = tiled_matmul(64, 64, 64)
    backend = CBackend()
    backend.prebuild(func, Target(), constants=frozenset({1}))
    assert build_harness(func, Target(), frozenset({1})).exists()
    seconds = backend.measure(func, Target(), min_seconds=0.01, constants=frozenset({1}))
    assert 1e-8 < seconds < 1e-2


def _dense_layers(layers: int, m: int, k: int, n: int) -> Graph:
    """``layers`` products x @ W_l + b_l of one input with different weights:
    one tuning task, whose weights a run streams from memory one after another."""
    rng = np.random.default_rng(0)
    g = Graph("dense")
    x = g.add_input("x", TensorType((m, k), DType.float32))
    outputs = []
    for i in range(layers):
        w = g.add_constant(rng.standard_normal((k, n)).astype(np.float32) * 0.03, f"w{i}")
        b = g.add_constant(rng.standard_normal(n).astype(np.float32), f"b{i}")
        outputs.append(g.op("Add", g.op("MatMul", x, w), b))
    g.set_outputs(outputs)
    return g


def _in_model(comp: Compilation, task: str, x: np.ndarray) -> float:
    """Mean time of the task's kernels in the built model, best of three profiles."""
    model = comp.build()
    names = {k.name for k in comp.kernels if k.task_key == task}
    profiles = [model.profile({"x": x}, repeat=10) for _ in range(3)]
    return min(statistics.mean(t.seconds for t in p if t.kernel in names) for p in profiles)


@pytest.mark.sme
@pytest.mark.skipif(not host_supports("sme"), reason="this CPU has no SME")
def test_measure_ranks_weight_configs_as_the_model_does() -> None:
    # GPT-2's QKV projection: the default (one 128 x 768 tile per thread) and
    # many small tiles. Timed alone on one warm weight, the default once
    # looked twice as slow as the small tiles; in the model it is faster.
    target = Target(isa="sme", num_threads=4)
    graph = _dense_layers(6, 128, 768, 2304)
    default = lower(graph, target=target, tuned=False)
    (task,) = {k.task_key for k in default.kernels if k.task_key}
    kernel = next(k for k in default.kernels if k.task_key == task)
    assert kernel.config is not None
    assert kernel.lowered is not None
    small: Mapping[str, KnobValue] = {**kernel.config, "micro_m": 32, "micro_n": 64}
    log = BestConfigs([Record(target.key, task, small, 1e-3)])
    tiled = lower(graph, target=target, tuning_log=log)
    x = np.random.default_rng(1).standard_normal((128, 768)).astype(np.float32)
    inputs = kernel.lowered.group.inputs
    constants = frozenset(i for i, v in enumerate(inputs) if v.data is not None)
    assert constants == {1, 2}
    backend = CBackend()
    measured, modelled = [], []
    for comp in (default, tiled):
        func = next(k.func for k in comp.kernels if k.task_key == task)
        measured.append(backend.measure(func, target, constants=constants))
        modelled.append(_in_model(comp, task, x))
    assert (measured[0] < measured[1]) == (modelled[0] < modelled[1]), (measured, modelled)
