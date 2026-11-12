"""``tessera bench models``: the four benchmark models against the baselines.

For every model and system (ONNX Runtime on the CPU, PyTorch eager on the
CPU and on MPS, Tessera on each requested target, tuned and untuned) this
measures latency (median of at least 50 runs after a warmup, with the 10th
and 90th percentiles) on the model's fixed input, and agreement with ONNX
Runtime: the largest absolute and relative difference of the outputs and,
for the image models, top-1 agreement over ``CLASSIFIER_SAMPLES`` inputs
(the fixed one and seeded random ones).

The ablations compile each model again with one part of the pipeline
turned off and report what changes: fusion (kernel count, bytes moved,
latency), the graph rewrites (``optimize=False``: no folding, no pattern
rewrites, BatchNorm decomposed instead of folded), memory planning (the
arena against the sum of the intermediate tensors it holds) and tuning
(template defaults against the tuning logs).

"Bytes moved" is the sum, over kernel launches, of the bytes of every
argument: what the kernels would read and write if each touched its
arguments once. It is a lower bound of memory traffic that does not depend
on the hardware, and fusion lowers it by keeping intermediates out of
memory.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from tessera.bench.baselines import Runner, onnxruntime_cpu, torch_available, torch_eager
from tessera.bench.report import markdown_table
from tessera.bench.stats import Agreement, Latency, agreement, time_runs
from tessera.bench.zoo import ModelSpec
from tessera.driver import Compilation, lower
from tessera.ir.graph import NDArray
from tessera.plan import StorageKind

CLASSIFIER_SAMPLES = 16
WARMUP = 10
RUNS = 50
MIB = 2**20
ROW_SECONDS = 60.0  # timing budget of one row: slower rows get fewer runs (at least 3)

# Ablation variants of the untuned pipeline: (fuse, optimize, schedule) for lower().
VARIANTS: dict[str, tuple[bool, bool, bool]] = {
    "full": (True, True, True),
    "fusion off": (False, True, True),
    "graph rewrites off": (True, False, True),
    "no schedule": (True, True, False),
}


@dataclass
class SystemResult:
    """One system on one model."""

    system: str
    latency: Latency | None = None
    agreement: Agreement | None = None
    threads: int | None = None
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "latency": self.latency.as_dict() if self.latency else None,
            "agreement": self.agreement.as_dict() if self.agreement else None,
            "threads": self.threads,
            "error": self.error,
            **self.details,
        }


@dataclass
class ModelResult:
    model: str
    systems: list[SystemResult]
    ablations: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelResult:
        """Read back what ``as_dict`` wrote (for pages that merge earlier runs)."""
        systems = []
        for s in data["systems"]:
            lat, agr = s.get("latency"), s.get("agreement")
            known = {"system", "latency", "agreement", "threads", "error"}
            systems.append(
                SystemResult(
                    s["system"],
                    Latency(**lat) if lat else None,
                    Agreement(**agr) if agr else None,
                    s.get("threads"),
                    s.get("error"),
                    {k: v for k, v in s.items() if k not in known},
                )
            )
        return cls(data["model"], systems, data.get("ablations", {}))

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "systems": [s.as_dict() for s in self.systems],
            "ablations": self.ablations,
        }


def samples(spec: ModelSpec) -> list[dict[str, NDArray]]:
    """The inputs agreement is measured on: the fixed one, plus random ones for classifiers."""
    fixed = spec.sample_inputs()
    if not spec.classifier:
        return [fixed]
    return [fixed, *(spec.random_inputs(seed) for seed in range(1, CLASSIFIER_SAMPLES))]


def measure_runner(
    runner: Runner,
    spec: ModelSpec,
    inputs: Sequence[Mapping[str, NDArray]],
    reference: list[dict[str, NDArray]],
) -> SystemResult:
    """Latency on the first input and agreement with ``reference`` on all of them."""
    outputs = [runner.outputs(x) for x in inputs]
    latency = time_runs(runner.prepare(inputs[0]), warmup=WARMUP, runs=RUNS)
    return SystemResult(
        runner.name,
        latency,
        agreement(outputs, reference, classifier=spec.classifier),
        runner.threads,
    )


def measure_compilation(
    name: str,
    comp: Compilation,
    spec: ModelSpec,
    inputs: Sequence[Mapping[str, NDArray]],
    reference: list[dict[str, NDArray]],
) -> SystemResult:
    """Build ``comp``, then time it and compare its outputs like a baseline.

    A model that runs too slowly for ``RUNS`` runs within ``ROW_SECONDS`` (the
    unscheduled pipeline) gets as many runs as fit, at least 3; the latency
    records how many.
    """
    compiled = comp.build()
    outputs = [compiled.run(x) for x in inputs]
    first = compiled.benchmark(inputs[0], warmup=0, repeat=1)[0]
    if first * (RUNS + WARMUP) <= ROW_SECONDS:
        seconds = compiled.benchmark(inputs[0], warmup=WARMUP, repeat=RUNS)
    else:
        seconds = compiled.benchmark(inputs[0], warmup=0, repeat=max(3, int(ROW_SECONDS / first)))
    return SystemResult(
        name,
        Latency.of(seconds),
        agreement(outputs, reference, classifier=spec.classifier),
        comp.target.threads if comp.target.kind == "c" else None,
        details=plan_stats(comp),
    )


def plan_stats(comp: Compilation) -> dict[str, Any]:
    """Kernel count, bytes moved and memory of a compilation (see the module docstring)."""
    moved = sum(ref.nbytes for launch in comp.plan.launches for ref in launch.args)
    held = {  # the tensors kernels write into the arena (views share their input's bytes)
        g.output.name: g.output.type.nbytes
        for g in comp.groups
        if not g.is_view and comp.memory.refs[g.output.name].kind is StorageKind.ARENA
    }
    return {
        "kernels": len(comp.plan.launches),
        "tuned_kernels": comp.tuned_kernels,
        "bytes_moved": moved,
        "arena_bytes": comp.memory.arena_bytes,
        "intermediate_bytes": sum(held.values()),
        "constant_bytes": int(comp.memory.constants.nbytes),
    }


def bench_model(
    spec: ModelSpec,
    *,
    targets: Sequence[str],
    baselines: Sequence[str] = ("onnxruntime-cpu", "pytorch-cpu", "pytorch-mps"),
    ablation_targets: Sequence[str] | None = None,
    variants: Sequence[str] = tuple(VARIANTS),
    progress: Callable[[str], None] = lambda message: None,
) -> ModelResult:
    """The baselines, Tessera tuned and untuned on each of ``targets``, and the
    ablation ``variants`` on each of ``ablation_targets`` (default: ``targets``)."""
    inputs = samples(spec)
    reference_runner = onnxruntime_cpu(spec.onnx_path())
    reference = [reference_runner.outputs(x) for x in inputs]
    results: list[SystemResult] = []

    def attempt(name: str, fn: Callable[[], SystemResult]) -> SystemResult:
        progress(f"{spec.name}: {name}")
        try:
            result = fn()
        except Exception as e:  # one failing system must not lose the others' numbers
            result = SystemResult(name, error=f"{type(e).__name__}: {e}")
        results.append(result)
        return result

    def baseline(name: str) -> SystemResult:
        runner = reference_runner
        if name.startswith("pytorch-"):
            runner = torch_eager(spec, name.removeprefix("pytorch-"))
        return measure_runner(runner, spec, inputs, reference)

    def tessera(label: str, target: str, tuned: bool) -> SystemResult:
        comp = lower(spec.onnx_path(), target=target, input_shapes=spec.input_shapes, tuned=tuned)
        return measure_compilation(label, comp, spec, inputs, reference)

    for name in baselines:
        if not name.startswith("pytorch-") or torch_available(name.removeprefix("pytorch-")):
            attempt(name, partial(baseline, name))
    for target in targets:
        for tuned in (True, False):
            label = f"tessera-{target}{'' if tuned else '-untuned'}"
            attempt(label, partial(tessera, label, target, tuned))
    found: dict[str, Any] = {}
    for target in targets if ablation_targets is None else ablation_targets:
        if variants:
            found[target] = ablate(spec, target, variants, inputs[:1], reference[:1], progress)
    return ModelResult(spec.name, results, found)


def ablate(
    spec: ModelSpec,
    target: str,
    variants: Sequence[str],
    inputs: Sequence[Mapping[str, NDArray]],
    reference: list[dict[str, NDArray]],
    progress: Callable[[str], None],
) -> dict[str, Any]:
    """Each variant of ``VARIANTS``, untuned, on the fixed input."""
    out: dict[str, Any] = {}
    for name in variants:
        progress(f"{spec.name}: {target} ablation, {name}")
        fuse, optimize, schedule = VARIANTS[name]
        try:
            comp = lower(
                spec.onnx_path(),
                target=target,
                input_shapes=spec.input_shapes,
                tuned=False,
                fuse=fuse,
                optimize=optimize,
                schedule=schedule,
            )
            out[name] = measure_compilation(name, comp, spec, inputs, reference).as_dict()
        except Exception as e:  # report the variant as failed, keep the others
            out[name] = {"system": name, "error": f"{type(e).__name__}: {e}"}
    return out


def markdown(results: Sequence[ModelResult]) -> str:
    """The tables of ``results/models.md`` (without the provenance header)."""
    parts: list[str] = []
    for r in results:
        rows: list[list[object]] = []
        base = next(
            (s.latency.median for s in r.systems if s.system == "onnxruntime-cpu" and s.latency),
            None,
        )
        for s in r.systems:
            if s.latency is None:
                rows.append([s.system, "failed", "", "", "", "", (s.error or "")[:80]])
                continue
            a = s.agreement
            speedup = f"{base / s.latency.median:.2f}x" if base else ""
            rows.append(
                [
                    s.system,
                    s.latency.ms(),
                    speedup,
                    f"{a.max_abs:.2e}" if a else "",
                    f"{a.max_rel:.2e}" if a else "",
                    f"{a.top1:.0%}" if a and a.top1 is not None else "",
                    _notes(s),
                ]
            )
        parts.append(
            f"### {r.model}\n\n"
            + markdown_table(
                [
                    "system",
                    "ms, median (p10 to p90)",
                    "vs ONNX Runtime",
                    "max abs err",
                    "max rel err",
                    "top-1",
                    "notes",
                ],
                rows,
            )
        )
        if r.ablations:
            parts.append(_ablation_table(r))
    return "\n\n".join(parts)


def _notes(s: SystemResult) -> str:
    notes = []
    if s.latency is not None and s.latency.runs < RUNS:
        notes.append(f"{s.latency.runs} runs")
    on_gpu = s.system.endswith("-mps") or "metal" in s.system
    if s.threads and not on_gpu:
        notes.append(f"{s.threads} threads")
    elif s.system == "onnxruntime-cpu":
        notes.append("default threads")
    if "kernels" in s.details:
        notes.append(f"{s.details['kernels']} kernels")
        if s.details.get("tuned_kernels"):
            notes.append(f"{s.details['tuned_kernels']} tuned")
    return ", ".join(notes)


def _ablation_table(r: ModelResult) -> str:
    rows: list[list[object]] = []
    for target, variants in r.ablations.items():
        for name, v in variants.items():
            if v.get("error"):
                rows.append([target, name, "failed", "", "", "", "", v["error"][:80]])
                continue
            lat = v["latency"]
            rows.append(
                [
                    target,
                    name,
                    v["kernels"],
                    f"{v['bytes_moved'] / MIB:.1f}",
                    f"{v['arena_bytes'] / MIB:.1f}",
                    f"{v['intermediate_bytes'] / MIB:.1f}",
                    f"{lat['median'] * 1e3:.2f}",
                    lat["runs"],
                ]
            )
    return f"Ablations of {r.model} (untuned):\n\n" + markdown_table(
        [
            "target",
            "pipeline",
            "kernels",
            "bytes moved (MiB)",
            "arena (MiB)",
            "intermediates (MiB)",
            "ms (median)",
            "runs",
        ],
        rows,
    )
