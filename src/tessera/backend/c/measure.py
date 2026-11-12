"""Timing one kernel for the tuner.

The kernel is compiled into a small executable with a ``main`` that fills
every parameter with random data, calibrates how many calls make one sample
and prints the per-call time of each sample. Running it as a separate process
means a candidate that crashes or hangs costs a ``MeasureError``, not the
tuner. Executables are cached like models, so measuring the same function
twice compiles once.
"""

from __future__ import annotations

import signal
import statistics
import subprocess
from pathlib import Path

from tessera.backend.base import MeasureError
from tessera.backend.c import build as cbuild
from tessera.backend.c.codegen import CodegenError, c_type, generate_kernel
from tessera.backend.c.microkernels import header_text
from tessera.target import Target
from tessera.tir.stmt import PrimFunc

DEFAULT_TIMEOUT = 60.0


def harness_sources(func: PrimFunc, target: Target) -> dict[str, str]:
    """``harness.c`` (the kernel plus a timing ``main``) and the headers it includes."""
    code = generate_kernel(func, target.isa)
    lines = [
        f"/* Timing harness for {func.name}, c:{target.isa}, {target.threads} threads. */",
        "",
        f"#define TSR_NUM_THREADS {target.threads}",
        '#include "tessera_runtime.h"',
        *(f'#include "{h}"' for h in code.headers),
        "",
        code.text,
        "static uint64_t rng_state = 0x9E3779B97F4A7C15ull;",
        "",
        "/* xorshift64*: uniform in [-1, 1). */",
        "static float random_unit(void) {",
        "  rng_state ^= rng_state >> 12;",
        "  rng_state ^= rng_state << 25;",
        "  rng_state ^= rng_state >> 27;",
        "  uint64_t bits = rng_state * 0x2545F4914F6CDD1Dull;",
        "  return (float)(bits >> 40) * (2.0f / 16777216.0f) - 1.0f;",
        "}",
        "",
        "int main(int argc, char** argv) {",
        "  int repeat = argc > 1 ? atoi(argv[1]) : 5;",
        "  double min_seconds = argc > 2 ? atof(argv[2]) : 0.05;",
        "  if (repeat < 1) repeat = 1;",
    ]
    args = []
    for i, buf in enumerate(func.params):
        ctype = c_type(buf.dtype)
        name = f"p{i}"
        args.append(name)
        lines.append(f"  {ctype}* {name} = ({ctype}*)tsr_alloc({buf.numel * buf.dtype.itemsize});")
        if buf.dtype.is_float:
            # Integer and boolean parameters (indices, masks) stay zero: always in range.
            lines.append(f"  for (int64_t i = 0; i < {buf.numel}; ++i) {name}[i] = random_unit();")
        else:
            lines.append(f"  memset({name}, 0, {buf.numel * buf.dtype.itemsize});")
    call = f"{code.name}({', '.join(args)});"
    lines += [
        f"  {call}",
        "  double start = tsr_now();",
        f"  {call}",
        "  double once = tsr_now() - start;",
        "  double per_sample = min_seconds / repeat;",
        "  int64_t number = once > 0 ? (int64_t)(per_sample / once) + 1 : 1000;",
        "  if (number > 1000000) number = 1000000;",
        "  for (int r = 0; r < repeat; ++r) {",
        "    start = tsr_now();",
        f"    for (int64_t n = 0; n < number; ++n) {call}",
        '    printf("%.9e\\n", (tsr_now() - start) / (double)number);',
        "  }",
        "  return 0;",
        "}",
        "",
    ]
    files = {"harness.c": "\n".join(lines), "tessera_runtime.h": header_text("tessera_runtime.h")}
    for h in code.headers:
        files[h] = header_text(h)
    return files


def build_harness(func: PrimFunc, target: Target) -> Path:
    """Compile (or reuse) the timing executable of ``func``; ``MeasureError`` if
    it does not build."""
    try:
        files = harness_sources(func, target)
        return cbuild.build(files, "harness.c", target, kind="executable")
    except (CodegenError, cbuild.BuildError) as e:
        raise MeasureError(f"{func.name}: {e}") from e


def run_harness(
    path: Path, *, repeat: int = 5, min_seconds: float = 0.05, timeout: float = DEFAULT_TIMEOUT
) -> list[float]:
    """Run a timing executable; seconds per call of each sample."""
    try:
        result = subprocess.run(
            [str(path), str(repeat), repr(min_seconds)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        raise MeasureError(f"timed out after {timeout:g} s") from e
    if result.returncode < 0:
        name = signal.Signals(-result.returncode).name
        raise MeasureError(f"crashed with {name}: {result.stderr.strip()}")
    if result.returncode != 0:
        raise MeasureError(f"exited with {result.returncode}: {result.stderr.strip()}")
    try:
        samples = [float(line) for line in result.stdout.split()]
    except ValueError as e:
        raise MeasureError(f"unexpected output: {result.stdout[:200]!r}") from e
    if len(samples) != repeat:
        raise MeasureError(f"expected {repeat} samples, got {len(samples)}")
    return samples


def measure(
    func: PrimFunc,
    target: Target,
    *,
    repeat: int = 5,
    min_seconds: float = 0.05,
    timeout: float = DEFAULT_TIMEOUT,
) -> float:
    """Median seconds per call of ``func`` on random inputs.

    Each of the ``repeat`` samples runs the kernel enough times to take about
    ``min_seconds / repeat``. Raises ``MeasureError`` if the kernel does not
    build, crashes, or the run takes longer than ``timeout``.
    """
    path = build_harness(func, target)
    return statistics.median(
        run_harness(path, repeat=repeat, min_seconds=min_seconds, timeout=timeout)
    )
