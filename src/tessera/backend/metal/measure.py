"""Timing one kernel on the GPU, for tuning."""

from __future__ import annotations

import math
import statistics

import numpy as np

from tessera.backend.base import MeasureError
from tessera.backend.metal import runtime
from tessera.backend.metal.codegen import CodegenError, Kernel, generate, module_source
from tessera.dtype import DType
from tessera.tir.stmt import PrimFunc

# One measurement batch never launches the kernel more often than this.
MAX_BATCH = 4096
MAX_WARMUP = 20000
# Continuous GPU work that brings an idle GPU to its full clock (measured on an M4:
# about 30 ms; see docs/metal.md).
WARMUP_SECONDS = 0.05


def random_buffer(
    device: runtime.Device, dtype: DType, shape: tuple[int, ...], rng: np.random.Generator
) -> runtime.Buffer:
    """A buffer holding values every operator accepts: floats in [0.5, 1.5) (log and
    sqrt are defined, division is safe) and zeros for integers (valid indices)."""
    count = int(np.prod(shape, dtype=np.int64))
    buffer = device.buffer(count * dtype.itemsize)
    if dtype.is_float:
        buffer.view(dtype, (count,))[:] = rng.random(count, dtype=np.float32) + 0.5
    return buffer


def measure_kernel(
    kernel: Kernel,
    *,
    repeat: int = 5,
    min_seconds: float = 0.05,
    warmup_seconds: float = WARMUP_SECONDS,
    timeout: float = 10.0,
    seed: int = 0,
) -> float:
    """Median GPU seconds of one launch of ``kernel``.

    The GPU first runs the kernel without pause for ``warmup_seconds``: an idle
    GPU runs at a low clock and only speeds up under sustained load, which would
    otherwise make small kernels look several times slower than they are. Then
    each of ``repeat`` measurements is one command buffer holding a batch of
    launches separated by barriers, timed by the GPU; the batch is sized so that
    all measurements together take about ``min_seconds``, which hides the cost
    of starting a command buffer behind many launches.
    """
    device = runtime.device()
    library = device.compile(module_source([kernel]))
    pipeline = library.pipeline(kernel.name)
    rng = np.random.default_rng(seed)
    args = [(random_buffer(device, b.dtype, b.shape, rng), 0) for b in kernel.buffers]

    def batch(count: int) -> runtime.Program:
        program = device.program()
        for _ in range(count):
            program.add(pipeline, kernel.grid, kernel.threads, args, barrier=True)
        return program

    def gpu_seconds(count: int) -> float:
        return batch(count).run(1, timeout=timeout).gpu[0]

    first = gpu_seconds(1)
    if warmup_seconds > 0:
        gpu_seconds(min(MAX_WARMUP, max(1, math.ceil(warmup_seconds / max(first, 1e-7)))))

    # Calibrate: grow the batch until one batch fills its share of the time.
    target = min_seconds / max(repeat, 1)
    count = 1
    while True:
        seconds = gpu_seconds(count)
        if seconds >= target or count >= MAX_BATCH:
            break
        grow = count * 2 if seconds <= 0 else max(count * 2, int(count * target / seconds))
        count = min(MAX_BATCH, grow)
    times = batch(count).run(max(repeat, 1), timeout=timeout).gpu
    return statistics.median(times) / count


def measure(
    func: PrimFunc,
    *,
    repeat: int = 5,
    min_seconds: float = 0.05,
    timeout: float = 10.0,
) -> float:
    """Median GPU seconds of one call of ``func`` on random inputs.

    Raises ``MeasureError`` when code generation, MSL compilation or the run fails.
    """
    try:
        kernel = generate(func, max_threadgroup_memory=runtime.device().info.max_threadgroup_memory)
        return measure_kernel(kernel, repeat=repeat, min_seconds=min_seconds, timeout=timeout)
    except (CodegenError, runtime.MetalError) as error:
        raise MeasureError(f"{func.name}: {error}") from error
