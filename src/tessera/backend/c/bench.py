"""Kernel benchmarks of the C backend, written to ``results/cpu/``.

    uv run python -m tessera.backend.c.bench [--quick] [--out results/cpu]

Three studies, each saved as Markdown and JSON:

- ``gemm``: C = A B in fp32 for square sizes and the matrix products of a
  transformer layer, compiled through the matmul template for each ISA
  (generic, neon, sme) with one thread and with every performance core,
  against NumPy's matmul (Accelerate on macOS) with the same thread counts.
- ``conv``: every distinct convolution of ResNet-50 at 224 x 224, batch 1,
  NHWC, with its bias and ReLU, through the conv2d template.
- ``sme_threads``: the SME microkernel driven from 1 to 4 threads, and one
  SME thread working beside NEON threads, to show how the performance cores
  share one SME unit.

Every compiled kernel is timed by ``measure`` (median of 5 samples, each a
loop of calls lasting about ``min_seconds / 5``) on random inputs.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tessera import compute, passes
from tessera.backend.c import build as cbuild
from tessera.backend.c.measure import measure
from tessera.backend.c.microkernels import header_files, host_supports
from tessera.backend.c.templates import make_templates
from tessera.dtype import DType
from tessera.ir import Graph, TensorType
from tessera.ir.graph import Value
from tessera.target import CpuIsa, Target, _cpu_name, performance_cores
from tessera.tir.schedule import Schedule
from tessera.tir.stmt import PrimFunc
from tessera.tune.space import ApplyConfig, DefineConfig

SQUARE = (256, 512, 1024, 2048, 4096)
TRANSFORMER = ((128, 768, 768), (128, 768, 3072), (128, 3072, 768))

# Distinct convolutions of ResNet-50 v1.5 (torchvision) at 224 x 224:
# (name, input size, input channels, output channels, kernel, stride).
RESNET50 = (
    ("conv1", 224, 3, 64, 7, 2),
    ("layer1 1x1 reduce", 56, 64, 64, 1, 1),
    ("layer1 3x3", 56, 64, 64, 3, 1),
    ("layer1 1x1 expand", 56, 64, 256, 1, 1),
    ("layer1 1x1 reduce (256)", 56, 256, 64, 1, 1),
    ("layer2 1x1 reduce", 56, 256, 128, 1, 1),
    ("layer2 3x3 stride 2", 56, 128, 128, 3, 2),
    ("layer2 1x1 expand", 28, 128, 512, 1, 1),
    ("layer2 downsample", 56, 256, 512, 1, 2),
    ("layer2 1x1 reduce (512)", 28, 512, 128, 1, 1),
    ("layer2 3x3", 28, 128, 128, 3, 1),
    ("layer3 1x1 reduce", 28, 512, 256, 1, 1),
    ("layer3 3x3 stride 2", 28, 256, 256, 3, 2),
    ("layer3 1x1 expand", 14, 256, 1024, 1, 1),
    ("layer3 downsample", 28, 512, 1024, 1, 2),
    ("layer3 1x1 reduce (1024)", 14, 1024, 256, 1, 1),
    ("layer3 3x3", 14, 256, 256, 3, 1),
    ("layer4 1x1 reduce", 14, 1024, 512, 1, 1),
    ("layer4 3x3 stride 2", 14, 512, 512, 3, 2),
    ("layer4 1x1 expand", 7, 512, 2048, 1, 1),
    ("layer4 downsample", 14, 1024, 2048, 1, 2),
    ("layer4 1x1 reduce (2048)", 7, 2048, 512, 1, 1),
    ("layer4 3x3", 7, 512, 512, 3, 1),
)

F32 = DType.float32


@dataclass(frozen=True)
class Row:
    """One measurement."""

    study: str
    case: str
    variant: str  # e.g. "neon t4", "numpy t1"
    seconds: float
    gflops: float


def isas() -> list[CpuIsa]:
    """The ISAs this machine runs, generic first."""
    return [isa for isa in ("generic", "neon", "sme") if host_supports(isa)]


def scheduled(build: Callable[[Graph], Value], target: Target) -> PrimFunc:
    """The single kernel of a graph, scheduled by its template's defaults."""
    g = Graph("bench")
    g.set_outputs([build(g)])
    (group,) = [grp for grp in passes.fuse(g) if not grp.is_view]
    lowered = compute.lower_group(group)
    template = make_templates(target)[group.pattern]
    cfg = DefineConfig()
    template(Schedule(lowered.func), lowered, cfg)
    sch = Schedule(lowered.func)
    template(sch, lowered, ApplyConfig(cfg.defaults))
    return sch.func


def gemm_graph(m: int, n: int, k: int) -> Callable[[Graph], Value]:
    def build(g: Graph) -> Value:
        a = g.add_input("A", TensorType((m, k), F32))
        b = g.add_constant(np.zeros((k, n), np.float32), "B")
        return g.op("MatMul", a, b)

    return build


def conv_graph(size: int, c: int, o: int, k: int, stride: int) -> Callable[[Graph], Value]:
    def build(g: Graph) -> Value:
        x = g.add_input("X", TensorType((1, size, size, c), F32))
        w = g.add_constant(np.zeros((k, k, c, o), np.float32), "W")
        b = g.add_constant(np.zeros(o, np.float32), "b")
        pad = k // 2
        y = g.op(
            "Conv2d", x, w, b,
            strides=(stride, stride), pads=(pad, pad, pad, pad), dilations=(1, 1),
            groups=1, layout="NHWC",
        )  # fmt: skip
        return g.op("Relu", y)

    return build


def conv_flops(size: int, c: int, o: int, k: int, stride: int) -> float:
    out = (size + 2 * (k // 2) - k) // stride + 1
    return 2.0 * out * out * o * k * k * c


def _time(func: PrimFunc, target: Target, min_seconds: float) -> float:
    return measure(func, target, repeat=5, min_seconds=min_seconds, timeout=600)


def numpy_gemm(m: int, n: int, k: int, threads: int, min_seconds: float) -> float:
    """Median seconds of ``A @ B`` in a fresh process whose Accelerate (or
    OpenBLAS) uses ``threads`` threads."""
    code = (
        "import json, statistics, sys, time, numpy as np\n"
        f"a = np.random.rand({m}, {k}).astype(np.float32)\n"
        f"b = np.random.rand({k}, {n}).astype(np.float32)\n"
        "a @ b\n"
        "start = time.perf_counter(); a @ b; once = time.perf_counter() - start\n"
        f"number = max(1, int({min_seconds} / 5 / max(once, 1e-7)))\n"
        "samples = []\n"
        "for _ in range(5):\n"
        "    start = time.perf_counter()\n"
        "    for _ in range(number): a @ b\n"
        "    samples.append((time.perf_counter() - start) / number)\n"
        "print(json.dumps(statistics.median(samples)))\n"
    )
    env = dict(os.environ)
    for var in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        env[var] = str(threads)
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    return float(json.loads(out.stdout))


def gemm_study(quick: bool) -> list[Row]:
    shapes = [(s, s, s) for s in (SQUARE[:2] if quick else SQUARE)]
    shapes += list(TRANSFORMER[:1] if quick else TRANSFORMER)
    threads = [1, performance_cores()]
    rows = []
    for m, n, k in shapes:
        case = f"{m}x{n}x{k}"
        flops = 2.0 * m * n * k
        budget = 0.2 if quick else 1.0
        for t in threads:
            seconds = numpy_gemm(m, n, k, t, budget)
            rows.append(Row("gemm", case, f"numpy t{t}", seconds, flops / seconds * 1e-9))
        for isa in isas():
            for t in threads:
                target = Target(isa=isa, num_threads=t)
                seconds = _time(scheduled(gemm_graph(m, n, k), target), target, budget)
                rows.append(Row("gemm", case, f"{isa} t{t}", seconds, flops / seconds * 1e-9))
                print(rows[-1], flush=True)
    return rows


def conv_study(quick: bool) -> list[Row]:
    layers = RESNET50[:4] if quick else RESNET50
    rows = []
    for name, size, c, o, k, stride in layers:
        flops = conv_flops(size, c, o, k, stride)
        for isa in isas():
            for t in (1, performance_cores()):
                target = Target(isa=isa, num_threads=t)
                func = scheduled(conv_graph(size, c, o, k, stride), target)
                seconds = _time(func, target, 0.1 if quick else 0.5)
                rows.append(Row("conv", name, f"{isa} t{t}", seconds, flops / seconds * 1e-9))
                print(rows[-1], flush=True)
    return rows


SME_THREADS_SOURCE = r"""
#define TSR_NUM_THREADS 4
#include "tessera_runtime.h"
#include "tessera_neon.h"
#include "tessera_sme.h"

typedef struct {
  int64_t m, n, k;
  const float *a, *b;
  float* c;
  int parts;
  double sme_share; /* hybrid: rows of part 0, computed with SME */
  int mode;         /* 0: every part SME, 1: every part NEON, 2: hybrid */
} job_t;

static void work(void* args, int64_t begin, int64_t end) {
  const job_t* j = (const job_t*)args;
  for (int64_t p = begin; p < end; ++p) {
    int64_t lo = j->m * p / j->parts, hi = j->m * (p + 1) / j->parts;
    if (j->mode == 2) {
      int64_t split = (int64_t)(j->m * j->sme_share) / 32 * 32, rest = j->m - split;
      lo = p == 0 ? 0 : split + rest * (p - 1) / (j->parts - 1);
      hi = p == 0 ? split : split + rest * p / (j->parts - 1);
    }
    if (hi <= lo) continue;
    if (j->mode == 0 || (j->mode == 2 && p == 0)) {
      tsr_sme_sgemm(hi - lo, j->n, j->k, j->a + lo * j->k, j->k, j->b, j->n, j->c + lo * j->n,
                    j->n, 0);
    } else {
      tsr_neon_sgemm(hi - lo, j->n, j->k, j->a + lo * j->k, j->k, j->b, j->n, j->c + lo * j->n,
                     j->n, 0);
    }
  }
}

static double best_of(job_t* j, int threads, int repeat) {
  double best = 1e30;
  for (int r = 0; r < repeat; ++r) {
    double start = tsr_now();
    tsr_parallel_for(0, threads, work, j);
    double t = tsr_now() - start;
    if (t < best) best = t;
  }
  return best;
}

int main(int argc, char** argv) {
  int64_t m = atol(argv[1]), n = atol(argv[2]), k = atol(argv[3]);
  int repeat = atoi(argv[4]);
  float* a = (float*)tsr_alloc((size_t)(m * k * 4));
  float* b = (float*)tsr_alloc((size_t)(k * n * 4));
  float* c = (float*)tsr_alloc((size_t)(m * n * 4));
  for (int64_t i = 0; i < m * k; ++i) a[i] = (float)(i % 13) * 0.1f;
  for (int64_t i = 0; i < k * n; ++i) b[i] = (float)(i % 7) * 0.1f;
  for (int t = 1; t <= 4; ++t) {
    job_t sme = {m, n, k, a, b, c, t, 0.0, 0}, neon = {m, n, k, a, b, c, t, 0.0, 1};
    printf("sme t%d %.9e\n", t, best_of(&sme, t, repeat));
    printf("neon t%d %.9e\n", t, best_of(&neon, t, repeat));
  }
  for (int s = 70; s <= 95; s += 5) {
    job_t hybrid = {m, n, k, a, b, c, 4, s / 100.0, 2};
    printf("sme t1 + neon t3, %d%% of rows on sme %.9e\n", s, best_of(&hybrid, 4, repeat));
  }
  return 0;
}
"""


def sme_threads_study(quick: bool) -> list[Row]:
    if not host_supports("sme"):
        return []
    files = {
        "sme_threads.c": SME_THREADS_SOURCE,
        **header_files(["tessera_sme.h", "tessera_neon.h"]),
    }
    binary = cbuild.build(files, "sme_threads.c", Target(isa="sme"), kind="executable")
    rows = []
    for m, n, k in [(1024, 1024, 1024), (128, 3072, 768)][: 1 if quick else 2]:
        out = subprocess.run(
            [str(binary), str(m), str(n), str(k), "3" if quick else "15"],
            capture_output=True, text=True, check=True, timeout=600,
        )  # fmt: skip
        for line in out.stdout.splitlines():
            label, seconds = line.rsplit(" ", 1)
            s = float(seconds)
            rows.append(Row("sme_threads", f"{m}x{n}x{k}", label, s, 2.0 * m * n * k / s * 1e-9))
    return rows


def machine() -> dict[str, Any]:
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
    return {
        "cpu": _cpu_name(),
        "performance_cores": performance_cores(),
        "system": f"{platform.system()} {platform.release()}",
        "compiler": cbuild.compiler_version("clang"),
        "numpy": np.__version__,
        "load_average": load,
        "date": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }


def markdown(title: str, rows: Sequence[Row], info: dict[str, Any], note: str) -> str:
    """A table with one row per case and one column per variant (GFLOPS)."""
    cases = list(dict.fromkeys(r.case for r in rows))
    variants = list(dict.fromkeys(r.variant for r in rows))
    value = {(r.case, r.variant): r.gflops for r in rows}
    lines = [f"# {title}", "", note, ""]
    lines.append(
        f"Measured {info['date']} on {info['cpu']} ({info['performance_cores']} performance "
        f"cores), {info['system']}, {info['compiler']}, NumPy {info['numpy']}; load average "
        f"{info['load_average']:.2f} at the start. Throughput in GFLOPS (higher is better)."
    )
    lines += ["", "| case | " + " | ".join(variants) + " |"]
    lines.append("|---|" + "---:|" * len(variants))
    for case in cases:
        cells = [f"{value[case, v]:.0f}" if (case, v) in value else "" for v in variants]
        lines.append(f"| {case} | " + " | ".join(cells) + " |")
    lines += ["", "Reproduce: `uv run python -m tessera.backend.c.bench`.", ""]
    return "\n".join(lines)


NOTES = {
    "gemm": (
        "fp32 C = A B with A, B and C row-major. `generic`, `neon` and `sme` are the "
        "kernels the matmul template generates for those targets; `numpy` is "
        "`numpy.matmul`, which calls Accelerate's sgemm on macOS. `t1` is one thread, "
        "`t4` every performance core."
    ),
    "conv": (
        "The distinct convolutions of ResNet-50 at 224 x 224, batch 1, NHWC input and "
        "HWIO weights, each with its bias and ReLU, as the conv2d template schedules "
        "them for each target."
    ),
    "sme_threads": (
        "The SME GEMM microkernel called from one to four threads, each on its own band "
        "of rows of C, against the NEON microkernel on the same bands; then one thread "
        "running SME on the given share of the rows while three run NEON on the rest."
    ),
}

TITLES = {
    "gemm": "GEMM throughput",
    "conv": "ResNet-50 convolutions",
    "sme_threads": "SME and threads",
}


def main(argv: Sequence[str] | None = None) -> None:
    """Run the studies and write ``<out>/<study>.md`` and ``.json``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="a few small cases only")
    parser.add_argument("--out", type=Path, default=Path("results/cpu"))
    parser.add_argument("--only", choices=sorted(TITLES), action="append")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    studies = {"gemm": gemm_study, "conv": conv_study, "sme_threads": sme_threads_study}
    for name in args.only or list(studies):
        info = machine()
        rows = studies[name](args.quick)
        if not rows:
            continue
        payload = {"machine": info, "rows": [asdict(r) for r in rows]}
        (args.out / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
        (args.out / f"{name}.md").write_text(markdown(TITLES[name], rows, info, NOTES[name]))
        print(f"wrote {args.out / name}.md", flush=True)


if __name__ == "__main__":
    main()
