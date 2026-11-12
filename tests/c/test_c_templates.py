"""Every CPU template, on every pattern, at its default and at random points of
its space: interpreted and compiled, the result matches NumPy."""

from __future__ import annotations

import random
from collections.abc import Callable

import numpy as np
import pytest
from c_workloads import Workload, elementwise, injective, layernorm, matmul, pool, reduce, softmax
from c_workloads import conv2d as conv2d_workload

from tessera import tir
from tessera.backend.base import Template, get_backend
from tessera.backend.c import run_kernel
from tessera.plan import PATTERNS
from tessera.target import Target
from tessera.tir.schedule import Schedule
from tessera.tune.space import ApplyConfig, ConfigSpace, DefineConfig, KnobValue

WORKLOADS: dict[str, Callable[[], Workload]] = {
    "matmul": lambda: matmul(12, 32, 20),
    "matmul_batched": lambda: matmul(6, 16, 9, batch=3, epilogue=False),
    "matmul_odd": lambda: matmul(7, 10, 5),
    "conv2d_3x3": lambda: conv2d_workload(1, 6, 8, 4, 16),
    "conv2d_strided": lambda: conv2d_workload(2, 9, 9, 3, 8, (7, 7), (2, 2), (3, 3, 3, 3)),
    "conv2d_1x1": lambda: conv2d_workload(1, 4, 4, 8, 32, (1, 1), (1, 1), (0, 0, 0, 0)),
    "conv2d_dilated": lambda: conv2d_workload(
        1, 8, 7, 5, 12, (3, 3), (1, 2), (2, 1, 0, 2), (2, 1), epilogue=False
    ),
    "depthwise": lambda: conv2d_workload(1, 8, 8, 16, 16, strides=(2, 2), depthwise=True),
    "maxpool": lambda: pool("max", 1, 9, 9, 8),
    "avgpool": lambda: pool("avg", 2, 8, 8, 4, (2, 2), (2, 2), (0, 0, 0, 0)),
    "reduce_last": lambda: reduce("sum", (6, 40), (1,)),
    "reduce_middle": lambda: reduce("mean", (2, 5, 6, 16), (1, 2)),
    "reduce_max": lambda: reduce("max", (3, 4, 7), (2,)),
    "softmax": lambda: softmax(5, 48, scale=True),
    "layernorm": lambda: layernorm(4, 64),
    "transpose": lambda: injective(2, 3, 5, 8),
    "gelu": lambda: elementwise((4, 32)),
}


def _template(w: Workload, target: Target) -> Template:
    return get_backend("c").templates(target)[w.lowered.pattern]


def _space(w: Workload, target: Target) -> ConfigSpace:
    cfg = DefineConfig()
    _template(w, target)(Schedule(w.func), w.lowered, cfg)
    return ConfigSpace(cfg.knobs, cfg.defaults)


def _scheduled(w: Workload, target: Target, values: dict[str, KnobValue]) -> Schedule:
    sch = Schedule(w.func)
    _template(w, target)(sch, w.lowered, ApplyConfig(values))
    tir.verify(sch.func)
    return sch


def _check(w: Workload, sch: Schedule, target: Target, *, interpret: bool) -> None:
    expected = w.expected()
    if interpret:
        out = w.output()
        tir.interpret(sch.func, [*w.inputs, out])
        np.testing.assert_allclose(out, expected, rtol=w.tolerance, atol=w.tolerance)
    out = w.output()
    run_kernel(sch.func, [*w.inputs, out], target)
    np.testing.assert_allclose(out, expected, rtol=w.tolerance, atol=w.tolerance)


def test_every_pattern_has_a_template() -> None:
    assert set(get_backend("c").templates(Target())) == set(PATTERNS)


@pytest.mark.parametrize("name", sorted(WORKLOADS))
def test_default_schedule(name: str) -> None:
    w = WORKLOADS[name]()
    target = Target(isa="generic", num_threads=4)
    space = _space(w, target)
    _check(w, _scheduled(w, target, space.default()), target, interpret=True)


@pytest.mark.parametrize("name", sorted(WORKLOADS))
def test_random_schedules(name: str) -> None:
    w = WORKLOADS[name]()
    target = Target(isa="generic", num_threads=3)
    space = _space(w, target)
    rng = random.Random(name)
    points = {space.index_of(space.sample(rng)) for _ in range(6)}
    for index in sorted(points):
        _check(w, _scheduled(w, target, space.get(index)), target, interpret=False)
