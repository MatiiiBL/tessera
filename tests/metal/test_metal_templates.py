"""Every GPU template, on groups lowered from small graphs: the scheduled
function verifies, interprets to the graph interpreter's output, and computes
it on the GPU.

The shapes are ones the default tiles divide: the schedule cannot yet bind a
fused loop that a tile splits imperfectly (a known limit of its injectivity
test, being lifted).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from metal_funcs import gpu
from metal_workloads import (
    Workload,
    configs,
    conv2d,
    depthwise,
    injective,
    layernorm,
    matmul,
    pool,
    reduce,
    softmax,
    space,
)

import tessera.tir as tir
from tessera.backend.metal import backend
from tessera.backend.metal.codegen import generate
from tessera.plan import PATTERNS
from tessera.tir.schedule import Schedule, ScheduleError
from tessera.tune.space import ApplyConfig, KnobValue

TEMPLATES = backend().templates()

# Shapes whose default tiles divide them evenly (see the module docstring).
# A thread-private accumulator used by several thread loops (matmul, conv), or
# given its buffer before it moves under the thread loop (pool, mean), is refused
# by the schedule's per-primitive check; so is binding a mixed-radix split it
# cannot yet prove one-to-one (depthwise). Both are being lifted in the tir package.
WAITING = pytest.mark.xfail(
    raises=ScheduleError, strict=True, reason="schedule check being relaxed in tir"
)

WORKLOADS = [
    pytest.param("injective", injective, id="injective"),
    pytest.param("matmul", lambda: matmul(128, 96, 64), id="matmul", marks=WAITING),
    pytest.param(
        "matmul", lambda: matmul(64, 64, 32, epilogue=False), id="matmul-bare", marks=WAITING
    ),
    pytest.param("matmul", lambda: matmul(32, 32, 64, batch=2), id="matmul-batched", marks=WAITING),
    pytest.param("conv", lambda: conv2d((1, 8, 8, 16), 32, 3, pad=1), id="conv3x3", marks=WAITING),
    pytest.param(
        "conv",
        lambda: conv2d((1, 16, 16, 8), 32, 3, stride=2, pad=1),
        id="conv3x3-s2",
        marks=WAITING,
    ),
    pytest.param("conv", lambda: conv2d((1, 8, 8, 32), 64, 1), id="conv1x1", marks=WAITING),
    pytest.param("depthwise", lambda: depthwise((1, 16, 16, 32)), id="depthwise", marks=WAITING),
    pytest.param(
        "depthwise", lambda: depthwise((1, 16, 16, 32), stride=2), id="depthwise-s2", marks=WAITING
    ),
    pytest.param("maxpool", lambda: pool("MaxPool2d", (1, 16, 16, 8)), id="maxpool"),
    pytest.param("avgpool", lambda: pool("AvgPool2d", (1, 16, 16, 8)), id="avgpool", marks=WAITING),
    pytest.param("sum", lambda: reduce("ReduceSum", (8, 256), (1,)), id="sum"),
    pytest.param(
        "mean", lambda: reduce("ReduceMean", (2, 8, 8, 64), (1, 2)), id="mean", marks=WAITING
    ),
    pytest.param("max", lambda: reduce("ReduceMax", (4, 512), (1,), keepdims=0), id="max"),
    pytest.param("softmax", lambda: softmax(12, 128), id="softmax"),
    pytest.param("softmax", lambda: softmax(4, 1024), id="softmax-wide"),
    pytest.param("layernorm", lambda: layernorm(8, 256), id="layernorm"),
]


def test_every_pattern_has_a_template() -> None:
    assert set(TEMPLATES) == set(PATTERNS)


def scheduled(workload: Workload, values: dict[str, KnobValue]) -> Schedule:
    sch = Schedule(workload.func)
    TEMPLATES[workload.pattern](sch, workload.lowered, ApplyConfig(values))
    return sch


def check_interpreted(sch: Schedule, workload: Workload) -> None:
    tir.verify(sch.func)
    out = sch.func.params[-1]
    result = np.zeros(out.shape, np.float32)
    tir.interpret(sch.func, [*workload.inputs, result], check_bounds=True)
    np.testing.assert_allclose(result, workload.output, rtol=1e-4, atol=1e-4)


def check_gpu(sch: Schedule, workload: Workload) -> None:
    from tessera.backend.metal.executor import run_func

    out = sch.func.params[-1]
    result = np.zeros(out.shape, np.float32)
    run_func(sch.func, [*workload.inputs, result])
    np.testing.assert_allclose(result, workload.output, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(("name", "make"), WORKLOADS)
def test_default_schedule_interprets_to_the_reference(
    name: str, make: Callable[[], Workload]
) -> None:
    workload = make()
    sch = scheduled(workload, space(TEMPLATES[workload.pattern], workload).default())
    check_interpreted(sch, workload)
    kernel = generate(sch.func)
    assert kernel.num_threads > 1


@gpu
@pytest.mark.parametrize(("name", "make"), WORKLOADS)
def test_schedules_run_on_the_gpu(name: str, make: Callable[[], Workload]) -> None:
    workload = make()
    ran = 0
    for values in configs(TEMPLATES[workload.pattern], workload, samples=6, seed=len(name)):
        try:
            sch = scheduled(workload, values)
        except ScheduleError:
            continue  # a point of the space the template cannot apply (the tuner skips it)
        check_gpu(sch, workload)
        ran += 1
    assert ran >= 1


@pytest.mark.parametrize(("name", "make"), WORKLOADS[:6])
def test_random_points_interpret_to_the_reference(name: str, make: Callable[[], Workload]) -> None:
    workload = make()
    for values in configs(TEMPLATES[workload.pattern], workload, samples=3, seed=7):
        try:
            sch = scheduled(workload, values)
        except ScheduleError:
            continue
        check_interpreted(sch, workload)
