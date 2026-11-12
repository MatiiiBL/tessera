"""Helpers shared by the CPU schedule templates.

A template asks its ``Config`` for every decision, and asks every knob on
every call whatever the other knobs' values are: the search space is
recorded from one run with the defaults, and every point of it must answer
the same questions.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from math import prod

from tessera.backend.c.alloc import children, sub_exprs
from tessera.target import Target
from tessera.tir.expr import Var
from tessera.tir.schedule import Schedule, ScheduleError
from tessera.tir.stmt import BlockRealize, Buffer, BufferStore, IterKind, Stmt
from tessera.tune.space import Config, KnobValue

# Parallel loops below this much work (multiply-adds or elements touched) run
# serially by default: starting the pool costs a few microseconds.
PARALLEL_WORK = 1 << 15

# Innermost vector loops: clang turns 4-float NEON operations into longer runs
# by interleaving, so extents up to 64 are useful.
VECTOR_SIZES = (4, 8, 16, 32, 64)


@dataclass(frozen=True)
class CpuTarget:
    """What a template needs to know about the target."""

    isa: str
    threads: int

    @classmethod
    def of(cls, target: Target) -> CpuTarget:
        return cls(target.isa, target.threads)


def divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def nearest(options: Sequence[int], want: int) -> int:
    """The option closest to ``want`` (the larger one on a tie)."""
    return min(options, key=lambda v: (abs(v - want), -v))


def choose(cfg: Config, name: str, options: Sequence[KnobValue], default: KnobValue) -> KnobValue:
    """``cfg.choice`` with the options deduplicated in order."""
    unique: list[KnobValue] = []
    for o in options:
        if o not in unique:
            unique.append(o)
    return cfg.choice(name, unique, default=default)


def choose_int(cfg: Config, name: str, options: Sequence[int], default: int) -> int:
    value = choose(cfg, name, options, default)
    assert isinstance(value, int)
    return value


def choose_bool(cfg: Config, name: str, default: bool) -> bool:
    value = cfg.choice(name, [default, not default], default=default)
    assert isinstance(value, bool)
    return value


def vector_options(extent: int) -> list[int]:
    """Inner extents a loop of ``extent`` can be split into for vectorization:
    divisors that are vector sizes, or 1 (no vectorization)."""
    return [1, *(v for v in VECTOR_SIZES if extent % v == 0)]


def register_options(extent: int, most: int = 8) -> list[int]:
    """Unroll factors (divisors up to ``most``) for a register-blocked loop."""
    return [d for d in divisors(extent) if d <= most]


def walk(stmt: Stmt) -> Iterator[Stmt]:
    stack = [stmt]
    while stack:
        s = stack.pop()
        yield s
        stack.extend(reversed(children(s)))


def realize(sch: Schedule, block: str) -> BlockRealize:
    """The BlockRealize of ``block`` in the current function."""
    for s in walk(sch.func.body):
        if isinstance(s, BlockRealize) and s.block.name == block:
            return s
    raise ScheduleError(f"no block named {block!r}")


def loop_kinds(sch: Schedule, block: str) -> list[str]:
    """For each loop around ``block``: ``S`` if it feeds only spatial
    iteration variables, ``R`` if only reduce ones, ``X`` if both, ``N`` if
    none."""
    r = realize(sch, block)
    spatial: set[Var] = set()
    reduce: set[Var] = set()
    for iv, value in zip(r.block.iter_vars, r.iter_values, strict=True):
        found = {e for e in sub_exprs(value) if isinstance(e, Var)}
        (reduce if iv.kind is IterKind.REDUCE else spatial).update(found)
    out = []
    for loop in sch.get_loops(block):
        s, rd = loop in spatial, loop in reduce
        out.append("X" if s and rd else "S" if s else "R" if rd else "N")
    return out


def output_block(sch: Schedule) -> str:
    """The block that stores into the function's last parameter (its output)."""
    out = sch.func.params[-1]
    for name in reversed(sch.blocks()):
        if stores_to(sch, name, out):
            return name
    raise ScheduleError("no block writes the output")


def stores_to(sch: Schedule, block: str, buf: Buffer) -> bool:
    return any(isinstance(s, BufferStore) and s.buffer is buf for s in walk(realize(sch, block)))


def try_apply(fn: object, *args: object) -> bool:
    """Apply a primitive if the IR allows it; False if it raised ScheduleError."""
    try:
        fn(*args)  # type: ignore[operator]
    except ScheduleError:
        return False
    return True


def place_epilogue(sch: Schedule, epilogue: str | None, loop: Var | None) -> None:
    """Inline the epilogue into the block that produces its input; if that is
    not possible, compute it under ``loop`` (on the tile just produced); if
    neither is, give it a loop nest of its own."""
    if epilogue is None or try_apply(sch.reverse_compute_inline, epilogue):
        return
    if loop is not None and try_apply(sch.reverse_compute_at, epilogue, loop):
        return
    schedule_alone(sch, epilogue)


def schedule_alone(sch: Schedule, block: str) -> None:
    """A fixed schedule (no knobs, so the space does not depend on whether
    fusion succeeded) for a block left in its own nest: the innermost loop
    vectorized, the others fused and parallel when there is enough work."""
    loops = sch.get_loops(block)
    if not loops:
        return
    work = prod(sch.extent(v) for v in loops)
    inner = loops[-1]
    lanes = [v for v in vector_options(sch.extent(inner)) if v <= 16][-1]
    outer = list(loops[:-1])
    if lanes == sch.extent(inner):
        sch.vectorize(inner)
    elif lanes > 1:
        head, tail = sch.split(inner, [None, lanes])
        sch.vectorize(tail)
        outer.append(head)
    else:
        outer.append(inner)
    parallelize(sch, outer, work >= PARALLEL_WORK)


def parallel_knob(cfg: Config, work: int, name: str = "parallel") -> bool:
    """Whether to run the outer loops on the thread pool (by default when there
    is enough work to amortize starting it)."""
    return choose_bool(cfg, name, work >= PARALLEL_WORK)


def parallelize(sch: Schedule, loops: Sequence[Var], enabled: bool) -> Var | None:
    """Fuse ``loops`` (consecutive, perfectly nested) and, if ``enabled``, run
    the result on the thread pool. Returns the fused loop."""
    if not loops:
        return None
    fused = loops[0] if len(loops) == 1 else sch.fuse(*loops)
    if enabled and sch.extent(fused) > 1:
        sch.parallel(fused)
    return fused
