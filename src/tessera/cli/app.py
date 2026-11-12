"""The ``tessera`` command: compile, run, inspect, profile, tune and benchmark models."""

from __future__ import annotations

import enum
import json
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from tessera import driver
from tessera.ir import interpret, random_inputs
from tessera.ir.graph import NDArray

app = typer.Typer(
    name="tessera",
    help="A tensor compiler for Apple silicon: ONNX models to fused, tuned C and Metal kernels.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
console = Console()

ModelArg = Annotated[Path, typer.Argument(help="An ONNX model file.", exists=True, dir_okay=False)]
TargetOpt = Annotated[
    str, typer.Option("--target", "-t", help="host, c:generic, c:neon, c:sme or metal.")
]
ShapeOpt = Annotated[
    list[str] | None,
    typer.Option("--shape", help="Fix an input's shape: NAME=1,3,224,224 (repeatable)."),
]
LogOpt = Annotated[
    list[Path] | None,
    typer.Option("--log", help="Tuning logs (files or directories); default results/tuning."),
]
UntunedOpt = Annotated[bool, typer.Option("--untuned", help="Use template defaults only.")]
NoFuseOpt = Annotated[bool, typer.Option("--no-fuse", help="One kernel per operator.")]
NoRewritesOpt = Annotated[
    bool, typer.Option("--no-rewrites", help="Minimal graph pipeline (no folding or rewrites).")
]


class Stage(enum.StrEnum):
    graph = "graph"
    optimized = "optimized"
    fused = "fused"
    tir = "tir"
    code = "code"


class Strategy(enum.StrEnum):
    model = "model"
    random = "random"


def parse_shapes(specs: list[str] | None) -> dict[str, tuple[int, ...]] | None:
    """``["x=1,3,224,224"]`` to ``{"x": (1, 3, 224, 224)}``."""
    if not specs:
        return None
    shapes: dict[str, tuple[int, ...]] = {}
    for spec in specs:
        name, sep, dims = spec.partition("=")
        try:
            if not sep or not name:
                raise ValueError
            shapes[name] = tuple(int(d) for d in dims.split(","))
        except ValueError:
            raise typer.BadParameter(f"expected NAME=D0,D1,..., got {spec!r}") from None
    return shapes


def _lower(
    model: Path,
    target: str,
    shape: list[str] | None,
    log: list[Path] | None = None,
    *,
    untuned: bool = False,
    no_fuse: bool = False,
    no_rewrites: bool = False,
) -> driver.Compilation:
    return driver.lower(
        model,
        target=target,
        input_shapes=parse_shapes(shape),
        tuning_log=list(log) if log else None,
        fuse=not no_fuse,
        optimize=not no_rewrites,
        tuned=not untuned,
    )


def _summary(comp: driver.Compilation, seconds: float) -> Table:
    table = Table(show_header=False, box=None)
    table.add_row("target", f"{comp.target.kind}:{comp.target.isa} ({comp.target.key})")
    table.add_row(
        "graph", f"{len(comp.imported.nodes)} nodes imported, {len(comp.graph.nodes)} optimized"
    )
    views = sum(1 for g in comp.groups if g.is_view)
    copies = sum(1 for k in comp.kernels if k.lowered is None)
    table.add_row(
        "kernels",
        f"{len(comp.kernels)} ({copies} copies); {views} views share their input's storage",
    )
    table.add_row("tuned", f"{comp.tuned_kernels} of {len(comp.kernels)} kernels from tuning logs")
    table.add_row("arena", f"{comp.memory.arena_bytes / 1e6:.2f} MB")
    table.add_row("constants", f"{comp.memory.constants.nbytes / 1e6:.2f} MB")
    table.add_row("time", f"{seconds:.1f} s")
    return table


@app.command("compile")
def compile_command(
    model: ModelArg,
    target: TargetOpt = "host",
    shape: ShapeOpt = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write the generated sources here.")
    ] = None,
    log: LogOpt = None,
    untuned: UntunedOpt = False,
    no_fuse: NoFuseOpt = False,
    no_rewrites: NoRewritesOpt = False,
) -> None:
    """Compile MODEL and print what was built."""
    started = time.monotonic()
    comp = _lower(
        model, target, shape, log, untuned=untuned, no_fuse=no_fuse, no_rewrites=no_rewrites
    )
    compiled = comp.build()
    console.print(_summary(comp, time.monotonic() - started))
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        for name, text in compiled.sources().items():
            (output / name).write_text(text)
        plan = {
            "target": comp.target.key,
            "kernels": [
                {"name": k.name, "task": k.task_key, "config": k.config, "tuned": k.tuned}
                for k in comp.kernels
            ],
            "arena_bytes": comp.memory.arena_bytes,
        }
        (output / "kernels.json").write_text(json.dumps(plan, indent=2, default=list) + "\n")
        console.print(f"wrote {len(compiled.sources()) + 1} files to {output}")


@app.command("run")
def run_command(
    model: ModelArg,
    target: TargetOpt = "host",
    shape: ShapeOpt = None,
    check: Annotated[
        bool, typer.Option("--check", help="Compare with the interpreter (float64).")
    ] = False,
    tolerance: Annotated[
        float, typer.Option(help="Largest error relative to the output's largest magnitude.")
    ] = 1e-3,
    seed: Annotated[int, typer.Option(help="Seed of the random inputs.")] = 0,
    log: LogOpt = None,
    untuned: UntunedOpt = False,
) -> None:
    """Run MODEL once on random inputs; with --check, compare with the interpreter."""
    comp = _lower(model, target, shape, log, untuned=untuned)
    compiled = comp.build()
    inputs = random_inputs(comp.graph, seed=seed)
    outputs = compiled.run(inputs)
    reference = {}
    if check:
        computed = interpret(comp.graph, inputs, float64=True)
        reference = {comp.graph.io_names.get(n, n): v for n, v in computed.items()}
    table = Table(
        "output",
        "shape",
        "dtype",
        "mean",
        "max |x|",
        *(["max abs err", "max rel err"] if check else []),
    )
    worst = 0.0
    for name, value in outputs.items():
        row = [
            name,
            str(value.shape),
            str(value.dtype),
            f"{_mean(value):.4g}",
            f"{_absmax(value):.4g}",
        ]
        if check:
            abs_err, rel_err = _errors(value, reference[name])
            worst = max(worst, rel_err)
            row += [f"{abs_err:.3g}", f"{rel_err:.3g}"]
        table.add_row(*row)
    console.print(table)
    if check:
        ok = worst <= tolerance
        console.print(
            f"[{'green' if ok else 'red'}]{'match' if ok else 'MISMATCH'}[/]: "
            f"worst relative error {worst:.3g} (tolerance {tolerance:g})"
        )
        if not ok:
            raise typer.Exit(1)


@app.command("inspect")
def inspect_command(
    model: ModelArg,
    stage: Annotated[Stage, typer.Option("--stage", "-s", help="What to print.")] = Stage.optimized,
    group: Annotated[
        str | None, typer.Option("--group", "-g", help="Only this group (fused and tir).")
    ] = None,
    target: TargetOpt = "host",
    shape: ShapeOpt = None,
    log: LogOpt = None,
    untuned: UntunedOpt = False,
) -> None:
    """Print one stage of the compilation of MODEL."""
    comp = _lower(model, target, shape, log, untuned=untuned)
    if stage is Stage.graph:
        typer.echo(str(comp.imported))
    elif stage is Stage.optimized:
        typer.echo(str(comp.graph))
    elif stage is Stage.fused:
        table = Table("group", "pattern", "nodes", "inputs", "output")
        for g in comp.groups:
            if group is None or g.name == group:
                pattern = "view" if g.is_view else g.pattern
                nodes = " ".join(f"{n.op}{'*' if n is g.anchor else ''}" for n in g.nodes)
                table.add_row(g.name, pattern, nodes, str(len(g.inputs)), str(g.output.type))
        console.print(table)
    elif stage is Stage.tir:
        from tessera.tir import script

        for k in comp.kernels:
            if group is None or k.name == group:
                origin = (
                    "tuned" if k.tuned else "default" if k.config is not None else "unscheduled"
                )
                typer.echo(f"# {k.name}: {k.task_key or 'copy'}, {origin} config {k.config or {}}")
                for line in k.trace:
                    typer.echo(f"#   {line}")
                typer.echo(script(k.func))
    else:
        for name, text in comp.build().sources().items():
            typer.echo(f"// ==== {name}")
            typer.echo(text)


@app.command("profile")
def profile_command(
    model: ModelArg,
    target: TargetOpt = "host",
    shape: ShapeOpt = None,
    repeat: Annotated[int, typer.Option(help="Profiled runs.")] = 10,
    log: LogOpt = None,
    untuned: UntunedOpt = False,
) -> None:
    """Time every kernel of MODEL (median over the profiled runs)."""
    comp = _lower(model, target, shape, log, untuned=untuned)
    compiled = comp.build()
    times = compiled.profile(random_inputs(comp.graph), repeat=repeat)
    by_name = {k.name: k for k in comp.kernels}
    total = sum(t.seconds for t in times) or 1.0
    table = Table("kernel", "pattern", "config", "ms", "share")
    for t in sorted(times, key=lambda t: -t.seconds):
        k = by_name.get(t.kernel)
        copy = k is None or k.lowered is None or k.group is None
        pattern = "copy" if copy or k is None or k.group is None else k.group.pattern
        origin = "tuned" if k is not None and k.tuned else "default"
        table.add_row(
            t.kernel, pattern, origin, f"{t.seconds * 1e3:.3f}", f"{t.seconds / total:.1%}"
        )
    console.print(table)
    console.print(f"total {total * 1e3:.3f} ms over {len(times)} kernels")


@app.command("tune")
def tune_command(
    models: Annotated[
        list[Path], typer.Argument(help="ONNX models; their tasks share the budget.", exists=True)
    ],
    target: TargetOpt = "host",
    shape: ShapeOpt = None,
    trials: Annotated[
        int, typer.Option(help="Measurements in total, counting those already in the log.")
    ] = 1000,
    strategy: Annotated[Strategy, typer.Option(help="Search strategy.")] = Strategy.model,
    log: Annotated[
        Path | None, typer.Option("--log", help="The log to append to (and resume from).")
    ] = None,
    batch: Annotated[int, typer.Option(help="Measurements per round.")] = 16,
    time_budget: Annotated[float | None, typer.Option(help="Stop after this many seconds.")] = None,
    build_workers: Annotated[int, typer.Option(help="Parallel build processes.")] = 3,
    seed: Annotated[int, typer.Option(help="Seed of the search.")] = 0,
    validate_rounds: Annotated[
        int,
        typer.Option(
            help="Afterwards, measure each task's best configurations and its default again "
            "this many times, interleaved (0: no validation)."
        ),
    ] = 3,
    validate_top: Annotated[int, typer.Option(help="Configurations validated per task.")] = 3,
    new_only: Annotated[
        bool, typer.Option(help="Tune only the tasks that have no record in the log yet.")
    ] = False,
    model_rounds: Annotated[
        int,
        typer.Option(
            help="Finally, profile each model tuned and untuned this many times and keep, per "
            "task, whichever configuration is faster inside the model (0: skip)."
        ),
    ] = 3,
) -> None:
    """Tune every kernel of the MODELS on TARGET and append the measurements to a log.

    Tasks that several models share are tuned once, weighted by how many
    kernels use them. The default log is results/tuning/<target>.jsonl, which
    tessera.compile reads; running the same command again resumes it.
    """
    from tessera.tune.log import BestConfigs, TuningLog
    from tessera.tune.measure import LocalRunner
    from tessera.tune.search import EvolutionarySearch, RandomSearch
    from tessera.tune.space import config_json
    from tessera.tune.task import Task, merge_task
    from tessera.tune.tuner import Round, tune, validate

    by_key: dict[str, Task] = {}
    kernels = 0
    for model in models:
        comp = _lower(model, target, shape, untuned=True)
        kernels += len(comp.kernels)
        for task in comp.tasks.values():
            merge_task(by_key, task)
    tasks = list(by_key.values())
    if new_only:
        logged = {r.task for r in TuningLog(log or _default_log(comp)).records()}
        tasks = [t for t in tasks if t.key not in logged]
    if not tasks:
        console.print("nothing to tune: no kernel of these models has a template")
        return
    path = log if log is not None else _default_log(comp)
    console.print(
        f"{len(tasks)} tasks for {kernels} kernels of {len(models)} models; logging to {path}"
    )
    search = EvolutionarySearch() if strategy is Strategy.model else RandomSearch()

    def report(r: Round) -> None:
        best = r.state.best()
        best_text = f"{best[1] * 1e6:.1f} us" if best is not None else "none ran"
        failed = sum(1 for m in r.results if m.seconds is None)
        console.print(
            f"[{r.measured_total}/{r.trials}] {r.task.key}: best {best_text}"
            + (f", {failed} failed" if failed else "")
        )

    started = time.monotonic()
    with LocalRunner(build_workers=build_workers) as runner:
        states = tune(
            tasks,
            search,
            trials=trials,
            runner=runner,
            log=TuningLog(path),
            batch_size=batch,
            seed=seed,
            time_budget=time_budget,
            on_round=report,
        )
        if validate_rounds:
            console.print(
                f"validating the {validate_top} best configurations and the default of each "
                f"task, {validate_rounds} rounds"
            )
            validate(
                tasks, runner=runner, log=TuningLog(path), top=validate_top, rounds=validate_rounds
            )
        build_s, measure_s = runner.build_seconds, runner.measure_seconds
    if model_rounds:
        from tessera.tune.inmodel import validate_in_model

        console.print("validating the tuned configurations inside the models")
        verdicts = validate_in_model(
            models, target, TuningLog(path), rounds=model_rounds, progress=console.print
        )
        slower = sorted(k for k, (default, tuned) in verdicts.items() if tuned > default)
        console.print(
            f"{len(verdicts)} tasks compared in the models; the default stays for "
            f"{len(slower)} whose tuned configuration is slower there"
            + (f": {', '.join(slower)}" if slower else "")
        )
    records = TuningLog(path).records()
    chosen = BestConfigs(records)
    table = Table("task", "groups", "trials", "default us", "best us", "speedup")
    for state in states:
        tuned = state.task
        default_key = _default_key(state)
        default_times = [
            r.seconds
            for r in records
            if r.task == tuned.key
            and r.target == tuned.target_key
            and r.seconds is not None
            and config_json(r.config) == default_key
        ]
        default = min(default_times) if default_times else None
        record = chosen.lookup(tuned.target_key, tuned.key)
        best = (dict(record.config), record.seconds) if record and record.seconds else None
        speed = f"{default / best[1]:.2f}x" if default and best else "-"
        table.add_row(
            state.task.key,
            str(getattr(state.task, "weight", 1)),
            str(len(state.measured)),
            f"{default * 1e6:.1f}" if default else "-",
            f"{best[1] * 1e6:.1f}" if best else "-",
            speed,
        )
    console.print(table)
    console.print(
        f"{time.monotonic() - started:.0f} s "
        f"({build_s:.0f} s building, {measure_s:.0f} s measuring)"
    )


class Suite(enum.StrEnum):
    models = "models"
    ops = "ops"
    tuning = "tuning"
    all = "all"


@app.command("bench")
def bench_command(
    suite: Annotated[Suite, typer.Argument(help="What to measure.")],
    target: Annotated[
        list[str] | None,
        typer.Option("--target", "-t", help="Tessera targets (default: the host's C and Metal)."),
    ] = None,
    model: Annotated[
        list[str] | None, typer.Option("--model", "-m", help="Only these models (models suite).")
    ] = None,
    baselines_only: Annotated[
        bool, typer.Option(help="Models suite: no Tessera rows (ablations still run).")
    ] = False,
    ablation: Annotated[
        list[str] | None,
        typer.Option(
            help="Models suite: ablation variants (full, fusion off, graph rewrites off, "
            "no schedule; default all)."
        ),
    ] = None,
    ablation_target: Annotated[
        list[str] | None,
        typer.Option(help="Models suite: targets of the ablations (default: --target)."),
    ] = None,
    trials: Annotated[int, typer.Option(help="Trials per search (tuning suite).")] = 256,
    seeds: Annotated[int, typer.Option(help="Seeds per strategy (tuning suite).")] = 5,
    real_tasks: Annotated[
        int,
        typer.Option(
            help="Tuning suite: also compare on this many of the heaviest tasks of ResNet-50 "
            "and BERT, measured on the first target (0: synthetic tasks only)."
        ),
    ] = 2,
    real_trials: Annotated[
        int, typer.Option(help="Tuning suite: trials per search on the real tasks.")
    ] = 96,
    provisional: Annotated[
        bool | None,
        typer.Option(
            "--provisional/--final",
            help="Mark the results (default: provisional when the machine is busy).",
        ),
    ] = None,
) -> None:
    """Measure and write results/<suite>.md and .json."""
    from tessera.bench import pages
    from tessera.bench.zoo import MODELS

    targets = list(target) if target else pages.default_targets()
    written = []
    if suite in (Suite.models, Suite.all):
        from tessera.bench.models import VARIANTS

        names = list(model) if model else list(MODELS)
        variants = list(ablation) if ablation else list(VARIANTS)
        unknown = sorted(set(variants) - set(VARIANTS))
        if unknown:
            raise typer.BadParameter(f"unknown ablations {unknown}; choose from {list(VARIANTS)}")
        written.append(
            pages.models_page(
                [] if baselines_only else targets,
                names=names,
                ablation_targets=list(ablation_target) if ablation_target else targets,
                variants=variants,
                provisional=provisional,
            )
        )
    if suite in (Suite.ops, Suite.all):
        written.append(pages.ops_page(targets, provisional=provisional))
    if suite in (Suite.tuning, Suite.all):
        real = []
        command = f"tessera bench tuning --trials {trials} --seeds {seeds}"
        if real_tasks:
            from tessera.bench.report import repo_root
            from tessera.bench.tuning import compare_real
            from tessera.bench.zoo import load_manifest

            specs = load_manifest()
            real = compare_real(
                [specs[name].onnx_path() for name in ("resnet50", "bert")],
                target=targets[0],
                tasks_per_model=real_tasks,
                trials=real_trials,
                seeds=range(seeds),
                log_dir=repo_root() / "data" / "tuning" / "spaces",
                on_run=lambda task, s, seed: console.print(f"{task}: {s}, seed {seed}"),
            )
            command += (
                f" --real-tasks {real_tasks} --real-trials {real_trials} --target {targets[0]}"
            )
        written.append(
            pages.tuning_page(
                real=real, trials=trials, seeds=seeds, provisional=provisional, command=command
            )
        )
    for path in written:
        console.print(f"wrote {path} and {path.with_suffix('.json')}")


def _default_key(state: object) -> str:
    from tessera.tune.search import TaskState
    from tessera.tune.space import config_json

    assert isinstance(state, TaskState)
    return config_json(state.task.space.default())


def _default_log(comp: driver.Compilation) -> Path:
    return driver.default_logs() / f"{_target_slug(comp)}.jsonl"


def _target_slug(comp: driver.Compilation) -> str:
    return "metal" if comp.target.kind == "metal" else f"c-{comp.target.isa}"


def _mean(a: NDArray) -> float:
    return float(np.mean(a)) if a.size else 0.0


def _absmax(a: NDArray) -> float:
    return float(np.max(np.abs(a))) if a.size else 0.0


def _errors(a: NDArray, b: NDArray) -> tuple[float, float]:
    diff = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
    worst = float(diff.max()) if diff.size else 0.0
    scale = _absmax(np.asarray(b, np.float64))
    return worst, worst / scale if scale > 0 else worst


def main() -> None:
    """Entry point of the ``tessera`` script."""
    app()
