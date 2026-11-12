"""Timing one kernel for the tuner.

The kernel is compiled into a small executable with a ``main`` that fills
every parameter with random data, calibrates how many calls make one sample
and prints the per-call time of each sample. Running it as a separate process
means a candidate that crashes or hangs costs a ``MeasureError``, not the
tuner. Executables are cached like models, so measuring the same function
twice compiles once.

The calls are timed as a model run makes them. Parameters that hold model
constants (``constants``) are prepared once before timing, as a model's
first run prepares them (the microkernels pack constant weights at their
first call and keep the packing), and the timed calls rotate through enough
copies of them to exceed the caches: a model reads each weight once per run,
from memory, not from the cache that the previous call left warm. Other
parameters stay warm, like the activations a kernel reads right after the
kernel before it wrote them.
"""

from __future__ import annotations

import signal
import statistics
import subprocess
from pathlib import Path

from tessera.backend.base import MeasureError
from tessera.backend.c import build as cbuild
from tessera.backend.c.codegen import CodegenError, c_type, generate_kernel
from tessera.backend.c.microkernels import header_files
from tessera.target import Target
from tessera.tir.stmt import PrimFunc

DEFAULT_TIMEOUT = 60.0


# The constant parameters of a harness span at least this many bytes in all
# their copies: more than the caches of the machines Tessera targets (the M4's
# performance cores share a 16 MiB L2). At most MAX_COPIES copies.
COLD_BYTES = 64 << 20
MAX_COPIES = 64


def constant_copies(func: PrimFunc, constants: frozenset[int]) -> int:
    """How many copies of the constant parameters the harness rotates through."""
    size = sum(func.params[i].numel * func.params[i].dtype.itemsize for i in constants)
    if size == 0:
        return 1
    return max(1, min(MAX_COPIES, -(-COLD_BYTES // size)))


def harness_sources(
    func: PrimFunc, target: Target, constants: frozenset[int] = frozenset()
) -> dict[str, str]:
    """``harness.c`` (the kernel plus a timing ``main``) and the headers it
    includes. ``constants``: the parameter indices that hold model constants."""
    if any(not 0 <= i < len(func.params) for i in constants):
        raise ValueError(f"constant indices {sorted(constants)} out of range for {func.name}")
    code = generate_kernel(func, target.isa)
    copies = constant_copies(func, constants)
    lines = [
        f"/* Timing harness for {func.name}, c:{target.isa}, {target.threads} threads. */",
        "",
        *cbuild.runtime_defines(target),
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
        "  tsr_prefer_performance_cores();",
        f"  enum {{ COPIES = {copies} }}; /* of each constant parameter */",
    ]
    args = []
    for i, buf in enumerate(func.params):
        ctype = c_type(buf.dtype)
        nbytes = buf.numel * buf.dtype.itemsize
        if i in constants:
            name = f"p{i}[c]"
            lines += [
                f"  {ctype}* p{i}[COPIES];",
                "  for (int c = 0; c < COPIES; ++c) {",
                f"    p{i}[c] = ({ctype}*)tsr_alloc({nbytes});",
            ]
            indent = "    "
        else:
            name = f"p{i}"
            lines.append(f"  {ctype}* {name} = ({ctype}*)tsr_alloc({nbytes});")
            indent = "  "
        if buf.dtype.is_float:
            # Integer and boolean parameters (indices, masks) stay zero: always in range.
            lines.append(
                f"{indent}for (int64_t i = 0; i < {buf.numel}; ++i) {name}[i] = random_unit();"
            )
        else:
            lines.append(f"{indent}memset({name}, 0, {nbytes});")
        if i in constants:
            lines.append("  }")
        args.append(name)
    call = f"{code.name}({', '.join(args)});"
    lines += [
        "  /* Prepare every copy of the constants, untimed, as a model's first run does. */",
        "  int c = 0;",
        f"  for (c = 0; c < COPIES; ++c) {call}",
        "  c = 0;",
        "  double start = tsr_now();",
        f"  {call}",
        "  double once = tsr_now() - start;",
        "  double per_sample = min_seconds / repeat;",
        "  int64_t number = once > 0 ? (int64_t)(per_sample / once) + 1 : 1000;",
        "  if (number > 1000000) number = 1000000;",
        "  for (int r = 0; r < repeat; ++r) {",
        "    start = tsr_now();",
        "    for (int64_t n = 0; n < number; ++n) {",
        "      c = c + 1 < COPIES ? c + 1 : 0;",
        f"      {call}",
        "    }",
        '    printf("%.9e\\n", (tsr_now() - start) / (double)number);',
        "  }",
        "  return 0;",
        "}",
        "",
    ]
    return {"harness.c": "\n".join(lines), **header_files(["tessera_runtime.h", *code.headers])}


def build_harness(func: PrimFunc, target: Target, constants: frozenset[int] = frozenset()) -> Path:
    """Compile (or reuse) the timing executable of ``func``; ``MeasureError`` if
    it does not build."""
    try:
        files = harness_sources(func, target, constants)
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
    constants: frozenset[int] = frozenset(),
) -> float:
    """Median seconds per call of ``func`` on random inputs.

    Each of the ``repeat`` samples runs the kernel enough times to take about
    ``min_seconds / repeat``. ``constants`` are the parameter indices that hold
    model constants: prepared before timing and read cold (see the module
    docstring). Raises ``MeasureError`` if the kernel does not build, crashes,
    or the run takes longer than ``timeout``.
    """
    path = build_harness(func, target, constants)
    return statistics.median(
        run_harness(path, repeat=repeat, min_seconds=min_seconds, timeout=timeout)
    )
