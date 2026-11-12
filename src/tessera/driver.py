"""``tessera.compile``: from an ONNX model to a ``CompiledModel``.

The driver runs the pipeline of docs/design.md, section 1:

    load -> optimize -> fuse -> plan memory -> lower each group
         -> apply its template with the best logged config (or the defaults)
         -> ExecutionPlan -> backend.build

``lower`` runs everything up to the ``ExecutionPlan`` and keeps every stage
in a ``Compilation``, which ``tessera inspect``, ``tessera tune`` and the
benchmarks read; ``compile`` builds it.

Kernels are named after their groups. A graph output that a kernel does not
write into its own output buffer (an output that is a view of a graph input,
a constant or another output) gets a copy kernel at the end of the plan, so
after a run every output is in its output buffer on every backend.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import onnx

from tessera import passes
from tessera.backend.base import CompiledModel, Template, get_backend
from tessera.frontend import onnx as onnx_frontend
from tessera.ir.graph import Graph, Value
from tessera.plan import (
    BufferRef,
    ExecutionPlan,
    FusedGroup,
    KernelLaunch,
    LoweredGroup,
    MemoryPlan,
    StorageKind,
)
from tessera.target import Target
from tessera.tir.expr import IntImm, Var
from tessera.tir.schedule import Schedule, ScheduleError
from tessera.tir.stmt import (
    TRUE,
    Block,
    BlockRealize,
    Buffer,
    BufferRegion,
    BufferStore,
    For,
    ForKind,
    IterVar,
    PrimFunc,
    Range,
)
from tessera.tune.log import BestConfigs
from tessera.tune.space import ApplyConfig, ConfigSpace, KnobValue
from tessera.tune.task import Task

type ModelLike = str | os.PathLike[str] | onnx.ModelProto | Graph
type LogsLike = str | os.PathLike[str] | Sequence[str | os.PathLike[str]] | BestConfigs | None


def default_logs() -> Path:
    """``$TESSERA_TUNING_LOGS``, or ``results/tuning`` of the checkout (the committed logs)."""
    checkout = Path(__file__).resolve().parents[2]
    return Path(os.environ.get("TESSERA_TUNING_LOGS", checkout / "results" / "tuning"))


@dataclass
class Kernel:
    """One launch of the plan: its function, its arguments and how it was made.

    ``group`` is None for the copies the driver adds (see the module docstring).
    """

    func: PrimFunc
    args: tuple[BufferRef, ...]  # in parameter order: the group's inputs, then its output
    group: FusedGroup | None = None
    lowered: LoweredGroup | None = None
    task_key: str | None = None
    config: dict[str, KnobValue] | None = None  # None: no template for this pattern
    tuned: bool = False  # the config came from a tuning log
    trace: list[str] = field(default_factory=list)  # the schedule's primitives

    @property
    def name(self) -> str:
        return self.func.name


@dataclass
class Compilation:
    """Every stage of compiling one model for one target."""

    target: Target
    imported: Graph
    graph: Graph  # after passes.optimize
    groups: list[FusedGroup]
    memory: MemoryPlan
    kernels: list[Kernel]
    plan: ExecutionPlan
    tasks: dict[str, Task]  # by task key, with weights = groups sharing it

    def build(self) -> CompiledModel:
        """Compile the plan with the target's backend."""
        return get_backend(self.target.kind).build(self.plan, self.target)

    @property
    def tuned_kernels(self) -> int:
        return sum(1 for k in self.kernels if k.tuned)


def resolve_target(target: str | Target) -> Target:
    """A ``Target`` from ``host``, ``c``, ``c:<isa>``, ``metal`` or a ``Target``."""
    return target if isinstance(target, Target) else Target.parse(target)


def load(model: ModelLike, input_shapes: Mapping[str, tuple[int, ...]] | None = None) -> Graph:
    """An imported graph (a ``Graph`` is used as is)."""
    if isinstance(model, Graph):
        return model
    source = model if isinstance(model, onnx.ModelProto) else Path(model)
    return onnx_frontend.load(source, input_shapes)


def lower(
    model: ModelLike,
    *,
    target: str | Target = "host",
    input_shapes: Mapping[str, tuple[int, ...]] | None = None,
    tuning_log: LogsLike = None,
    fuse: bool = True,
    optimize: bool = True,
    tuned: bool = True,
    lower_group: Callable[[FusedGroup], LoweredGroup] | None = None,
) -> Compilation:
    """Run the pipeline up to the ``ExecutionPlan``; see ``compile`` for the options."""
    tgt = resolve_target(target)
    imported = load(model, input_shapes)
    graph = passes.optimize(imported, fold=optimize)
    groups = passes.fuse(graph, enabled=fuse)
    passes.verify_groups(graph, groups)
    memory = passes.plan_memory(graph, groups)
    templates = get_backend(tgt.kind).templates(tgt)
    best = best_configs(tuning_log) if tuned else BestConfigs()
    lower_fn = lower_group if lower_group is not None else _lower_group

    refs = dict(memory.refs)
    taken = {g.name for g in groups}
    kernels: list[Kernel] = []
    tasks: dict[str, Task] = {}
    for group in groups:
        args = tuple(refs[v.name] for v in [*group.inputs, group.output])
        if group.is_view:
            if args[0] != args[-1]:  # the view's storage is not its input's: copy
                name = _unique(f"copy_{group.name}", taken)
                kernels.append(_copy(name, group.output, args[0], args[-1], group))
            continue
        lowered = lower_fn(group)
        template = templates.get(group.pattern)
        if template is None:
            func = _named(lowered.func, group)
            kernels.append(Kernel(func, args, group, lowered))
            continue
        task = Task.create(lowered, template, tgt)
        if task.key in tasks:
            tasks[task.key].weight += 1
            tasks[task.key].names.append(group.name)
        else:
            tasks[task.key] = task
        kernels.append(_schedule(group, lowered, args, template, task, best, tgt))

    for index, value in enumerate(graph.outputs):
        ref = refs[value.name]
        out = BufferRef(StorageKind.OUTPUT, 0, value.type.nbytes, index)
        if ref != out:  # an output written elsewhere (or an alias of another output)
            name = _unique(f"copy_{_identifier(value.name)}", taken)
            kernels.append(_copy(name, value, ref, out, None))
            refs[value.name] = out
    plan_memory = replace(memory, refs=refs)
    plan = ExecutionPlan(
        name=_identifier(graph.name),
        launches=[KernelLaunch(k.name, k.args) for k in kernels],
        memory=plan_memory,
        inputs=[(v.name, v.type) for v in graph.inputs],
        outputs=[(v.name, v.type) for v in graph.outputs],
        funcs={k.name: k.func for k in kernels},
    )
    return Compilation(tgt, imported, graph, groups, plan_memory, kernels, plan, tasks)


def compile(
    model: ModelLike,
    *,
    target: str | Target = "host",
    input_shapes: Mapping[str, tuple[int, ...]] | None = None,
    tuning_log: LogsLike = None,
    fuse: bool = True,
    optimize: bool = True,
    tuned: bool = True,
) -> CompiledModel:
    """Compile an ONNX model (a path, a ``ModelProto`` or an imported ``Graph``).

    ``target``: ``host`` (the best C target of this machine), ``c:generic``,
    ``c:neon``, ``c:sme`` or ``metal``. ``input_shapes`` fixes dynamic input
    dimensions by input name. ``tuning_log`` is a log file, a directory of
    logs, a list of them or a ``BestConfigs``; by default the committed logs
    in ``results/tuning``. Each kernel uses the fastest logged configuration
    of its task on this target, or the template's defaults. The ablation
    switches: ``fuse=False`` gives every operator its own kernel,
    ``optimize=False`` turns off the graph rewrites (the minimal pipeline:
    canonical operators and the NHWC layout only), and ``tuned=False``
    ignores the logs.
    """
    return lower(
        model,
        target=target,
        input_shapes=input_shapes,
        tuning_log=tuning_log,
        fuse=fuse,
        optimize=optimize,
        tuned=tuned,
    ).build()


def best_configs(tuning_log: LogsLike) -> BestConfigs:
    """Load the tuning logs ``compile`` was given (missing paths are skipped)."""
    if isinstance(tuning_log, BestConfigs):
        return tuning_log
    if tuning_log is None:
        paths: Sequence[str | os.PathLike[str]] = [default_logs()]
    elif isinstance(tuning_log, str | os.PathLike):
        paths = [tuning_log]
    else:
        paths = tuning_log
    return BestConfigs.load(*paths)


def apply_template(
    template: Template, lowered: LoweredGroup, config: Mapping[str, KnobValue]
) -> tuple[PrimFunc, list[str]]:
    """``lowered`` scheduled by ``template`` with ``config``, and the schedule's trace."""
    sch = Schedule(lowered.func)
    template(sch, lowered, ApplyConfig(config))
    return sch.func, list(sch.trace)


def _schedule(
    group: FusedGroup,
    lowered: LoweredGroup,
    args: tuple[BufferRef, ...],
    template: Template,
    task: Task,
    best: BestConfigs,
    target: Target,
) -> Kernel:
    record = best.lookup(target.key, task.key, accepts=lambda c: _in_space(c, task.space))
    configs = [(dict(record.config), True)] if record is not None else []
    configs.append((task.space.default(), False))
    for i, (config, tuned) in enumerate(configs):
        try:
            func, trace = apply_template(template, lowered, config)
        except ScheduleError:
            if i == len(configs) - 1:
                raise
            continue  # a logged config the template no longer accepts: use the defaults
        return Kernel(_named(func, group), args, group, lowered, task.key, config, tuned, trace)
    raise AssertionError("unreachable")


def _in_space(config: Mapping[str, KnobValue], space: ConfigSpace) -> bool:
    return set(config) == set(space.knobs) and all(config[k] in space.knobs[k] for k in space.knobs)


def _named(func: PrimFunc, group: FusedGroup) -> PrimFunc:
    return func if func.name == group.name else replace(func, name=group.name)


def _copy(
    name: str, value: Value, src: BufferRef, dst: BufferRef, group: FusedGroup | None
) -> Kernel:
    source = Buffer(f"{_identifier(value.name)}_in", (value.type.numel,), value.dtype)
    target = Buffer(_identifier(value.name), (value.type.numel,), value.dtype)
    return Kernel(copy_kernel(name, source, target), (src, dst), group)


def copy_kernel(name: str, src: Buffer, dst: Buffer) -> PrimFunc:
    """``dst[i] = src[i]`` over flat buffers of the same size."""
    i, vi = Var("i"), Var("vi")
    n = src.numel
    block = Block(
        "copy",
        (IterVar(vi, Range.of(n)),),
        (BufferRegion.full(src),),
        (BufferRegion.full(dst),),
        BufferStore(dst, src[vi], (vi,)),
    )
    loop = For(i, IntImm(0), IntImm(n), ForKind.SERIAL, BlockRealize((i,), TRUE, block))
    root = Block("root", (), (), (), loop)
    return PrimFunc(name, (src, dst), BlockRealize((), TRUE, root))


def _unique(name: str, taken: set[str]) -> str:
    candidate, i = name, 1
    while candidate in taken:
        candidate = f"{name}_{i}"
        i += 1
    taken.add(candidate)
    return candidate


def _identifier(name: str) -> str:
    out = "".join(c if c.isalnum() or c == "_" else "_" for c in name) or "model"
    return f"m_{out}" if out[0].isdigit() else out


def _lower_group(group: FusedGroup) -> LoweredGroup:
    import tessera.compute

    # Resolved at call time until tessera.compute provides lower_group.
    fn: Callable[[FusedGroup], LoweredGroup] = tessera.compute.__dict__["lower_group"]
    return fn(group)
