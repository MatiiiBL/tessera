"""Every GPU template, on groups lowered from small graphs: the scheduled
function verifies, interprets to the graph interpreter's output, and computes
it on the GPU."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from metal_funcs import gpu
from metal_workloads import (
    Workload,
    attention_scores,
    configs,
    conv2d,
    conv2d_pad_block,
    depthwise,
    injective,
    layernorm,
    matmul,
    output_projection,
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
WORKLOADS = [
    pytest.param("injective", injective, id="injective"),
    pytest.param("matmul", lambda: matmul(128, 96, 64), id="matmul"),
    pytest.param("matmul", lambda: matmul(33, 70, 19), id="matmul-odd"),
    pytest.param("matmul", lambda: matmul(64, 64, 32, epilogue=False), id="matmul-bare"),
    pytest.param("matmul", lambda: matmul(17, 40, 24, batch=3), id="matmul-batched"),
    pytest.param("matmul", lambda: matmul(1, 1000, 96), id="gemv"),
    pytest.param("matmul", lambda: matmul(32, 128, 48), id="matmul-simdgroup"),
    pytest.param("matmul", lambda: matmul(64, 64, 16, batch=3, epilogue=False), id="bmm-simdgroup"),
    pytest.param("matmul", lambda: output_projection(4, 32, 16, 48), id="matmul-operand-block"),
    pytest.param("matmul", lambda: attention_scores(2, 32, 16), id="matmul-operand-blocks"),
    pytest.param("conv", lambda: conv2d((1, 8, 8, 16), 32, 3, pad=1), id="conv3x3"),
    pytest.param("conv", lambda: conv2d((1, 7, 7, 24), 40, 3, pad=1), id="conv3x3-7x7"),
    pytest.param("conv", lambda: conv2d((2, 15, 15, 8), 24, 3, stride=2, pad=1), id="conv3x3-s2"),
    pytest.param("conv", lambda: conv2d((1, 14, 14, 32), 48, 1), id="conv1x1"),
    pytest.param("conv", lambda: conv2d((1, 32, 32, 16), 32, 1, stride=2), id="conv1x1-s2"),
    pytest.param("conv", lambda: conv2d((1, 23, 23, 3), 16, 7, stride=2, pad=3), id="conv7x7-stem"),
    pytest.param("conv", lambda: conv2d_pad_block((1, 8, 8, 32), 64, 3), id="conv-pad-block"),
    pytest.param("conv", lambda: conv2d_pad_block((1, 7, 7, 16), 24, 3), id="conv-pad-block-7x7"),
    pytest.param("depthwise", lambda: depthwise((1, 16, 16, 32)), id="depthwise"),
    pytest.param("depthwise", lambda: depthwise((1, 15, 15, 24), stride=2), id="depthwise-s2"),
    pytest.param("maxpool", lambda: pool("MaxPool2d", (1, 16, 16, 8)), id="maxpool"),
    pytest.param("maxpool", lambda: pool("MaxPool2d", (1, 13, 13, 5)), id="maxpool-odd"),
    pytest.param("avgpool", lambda: pool("AvgPool2d", (1, 11, 11, 6)), id="avgpool"),
    pytest.param("sum", lambda: reduce("ReduceSum", (8, 256), (1,)), id="sum"),
    pytest.param("sum", lambda: reduce("ReduceSum", (6, 200), (1,)), id="sum-odd"),
    pytest.param("mean", lambda: reduce("ReduceMean", (2, 7, 7, 64), (1, 2)), id="mean"),
    pytest.param("max", lambda: reduce("ReduceMax", (5, 37, 3), (1,), keepdims=0), id="max-middle"),
    pytest.param("softmax", lambda: softmax(12, 128), id="softmax"),
    pytest.param("softmax", lambda: softmax(7, 100), id="softmax-odd"),
    pytest.param("softmax", lambda: softmax(3, 1000), id="softmax-wide"),
    pytest.param("layernorm", lambda: layernorm(8, 256), id="layernorm"),
    pytest.param("layernorm", lambda: layernorm(5, 768), id="layernorm-bert"),
]


def test_simdgroup_matrices_are_the_default_for_whole_tiles() -> None:
    from metal_workloads import space as space_of_workload

    for workload in (matmul(32, 128, 48), matmul(64, 64, 16, batch=3, epilogue=False)):
        default = space_of_workload(TEMPLATES["matmul"], workload).default()
        assert default["strategy"] == "simdgroup"
        source = generate(scheduled(workload, default).func).source
        assert "simdgroup_multiply_accumulate" in source
    # 33 rows do not divide into 8-row tiles: scalar tiles instead.
    assert (
        space_of_workload(TEMPLATES["matmul"], matmul(33, 70, 19)).default()["strategy"] == "tiles"
    )


def test_convolutions_use_simdgroup_matrices_when_they_can() -> None:
    from metal_workloads import space as space_of_workload

    def strategy(workload: Workload) -> object:
        return space_of_workload(TEMPLATES["conv2d"], workload).default()["strategy"]

    # The padding as a block of its own, or no padding: the input is a plain load.
    assert strategy(conv2d_pad_block((1, 7, 7, 16), 24, 3)) == "simdgroup"
    assert strategy(conv2d((1, 14, 14, 32), 48, 1)) == "simdgroup"
    # A stride of 2 reads every other input row: the microkernel's rows are 2 apart.
    assert strategy(conv2d((1, 32, 32, 16), 32, 1, stride=2)) == "simdgroup"
    # Padding read through a select, or a stride of 2: scalar tiles.
    assert strategy(conv2d((1, 8, 8, 16), 32, 3, pad=1)) == "tiles"
    assert strategy(conv2d((2, 15, 15, 8), 24, 3, stride=2, pad=1)) == "tiles"
    source = generate(
        scheduled(
            conv2d_pad_block((1, 8, 8, 32), 64, 3),
            space_of_workload(
                TEMPLATES["conv2d"], conv2d_pad_block((1, 8, 8, 32), 64, 3)
            ).default(),
        ).func
    ).source
    assert "threadgroup float pad_shared[" in source
    assert "simdgroup_multiply_accumulate" in source


def test_operands_with_a_prologue_are_staged_unless_loaded_in_place() -> None:
    projection, scores = output_projection(4, 32, 16, 48), attention_scores(2, 32, 16)
    for workload in (projection, scores):
        assert space(TEMPLATES["matmul"], workload).default()["strategy"] == "simdgroup"
    # A transpose and reshape (k // 16, k % 16) is computed into threadgroup memory
    # once per element; a slice of the projection is a matrix loaded in place.
    source = generate(scheduled(projection, space(TEMPLATES["matmul"], projection).default()).func)
    assert "threadgroup float a_shared[" in source.source
    assert "simdgroup_multiply_accumulate" in source.source
    source = generate(scheduled(scores, space(TEMPLATES["matmul"], scores).default()).func)
    assert "threadgroup float kt_shared[" in source.source
    assert "q_shared" not in source.source
    # The scalar schedules read the operand through its prologue again.
    tiles = {**space(TEMPLATES["matmul"], projection).default(), "strategy": "tiles"}
    assert "a" not in scheduled(projection, tiles).blocks()


@pytest.mark.parametrize(
    "make",
    [lambda: output_projection(2, 16, 32, 40), lambda: attention_scores(3, 16, 32)],
    ids=["projection", "scores"],
)
def test_operand_block_points_interpret_to_the_reference(make: Callable[[], Workload]) -> None:
    workload = make()
    ran = 0
    for values in configs(TEMPLATES["matmul"], workload, samples=8, seed=3):
        try:
            sch = scheduled(workload, values)
        except ScheduleError:
            continue
        check_interpreted(sch, workload)
        ran += 1
    assert ran >= 3


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
