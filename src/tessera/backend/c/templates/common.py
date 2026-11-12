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
from tessera.backend.c.codegen import PARALLEL_THREADS, STACK_LIMIT
from tessera.target import Target
from tessera.tir.expr import Var
from tessera.tir.schedule import Schedule, ScheduleError
from tessera.tir.stmt import BlockRealize, Buffer, BufferStore, IterKind, Stmt
from tessera.tune.space import Config, KnobValue

# Parallel loops below this much work (multiply-adds or elements touched) run
# serially by default: starting the pool costs a few microseconds.
PARALLEL_WORK = 1 << 15

# Local copies (packed strips) are only made when they fit the stack: above
# STACK_LIMIT the generated code reaches them through a pointer, and clang
# then cannot prove they do not alias the accumulator, so it stops
# vectorizing the loops that use them.
PACK_LIMIT = STACK_LIMIT

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


def tile_options(extent: int, most: int) -> list[int]:
    """Tile sizes for a loop tensorized with edge tiles allowed: the divisors of
    ``extent`` up to ``most``, the powers of two below ``extent`` and
    ``extent`` itself (if not above ``most``)."""
    options = {d for d in divisors(extent) if d <= most}
    options |= {1 << e for e in range(4, most.bit_length()) if (1 << e) < extent}
    return sorted(options)


def balanced_tile(
    extent: int, others: int, threads: int, options: Sequence[int], widest: int = 256
) -> int:
    """A tile size for ``extent`` (with ``others`` tiles in the other
    directions) that gives every thread a whole number of tiles, the fewest
    that keep a tile at most ``widest``: wider tiles repack the operands less
    often, more tiles keep the threads busy."""
    per_thread = 1
    want = -(-extent * others // threads)
    while want > widest:
        per_thread *= 2
        want = -(-extent * others // (threads * per_thread))
    want = min(want, extent)
    fitting = [c for c in options if c <= want]
    return max(fitting) if fitting else min(options)


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


def iter_var_of(sch: Schedule, block: str, loop: Var) -> Var | None:
    """The iteration variable of ``block`` bound to exactly ``loop``."""
    r = realize(sch, block)
    for iv, value in zip(r.block.iter_vars, r.iter_values, strict=True):
        if value is loop:
            return iv.var
    return None


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


def read_index(sch: Schedule, block: str, *, uses: Sequence[Var], avoids: Var) -> int | None:
    """The position in ``block``'s reads of the one buffer whose region
    depends on every loop in ``uses`` and not on loop ``avoids`` (B of a
    matmul depends on the column and reduction loops, not the row loop)."""
    r = realize(sch, block)
    bound: dict[Var, set[Var]] = {}
    for iv, value in zip(r.block.iter_vars, r.iter_values, strict=True):
        bound[iv.var] = {e for e in sub_exprs(value) if isinstance(e, Var)}
    found = []
    for index, region in enumerate(r.block.reads):
        loops: set[Var] = set()
        for rng in region.region:
            for e in (*sub_exprs(rng.min), *sub_exprs(rng.extent)):
                if isinstance(e, Var):
                    loops |= bound.get(e, {e})
        if all(v in loops for v in uses) and avoids not in loops:
            found.append(index)
    return found[0] if len(found) == 1 else None


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


def vectorize_inside(sch: Schedule, block: str, tile: Var) -> None:
    """Vectorize the innermost loop of ``block`` below ``tile``, if it has one
    of more than one iteration (the nests an init or an epilogue gets under a
    tile)."""
    loops = sch.get_loops(block)
    inside = loops[loops.index(tile) + 1 :] if tile in loops else []
    if inside and sch.extent(inside[-1]) > 1:
        sch.vectorize(inside[-1])


def parallel_on(sch: Schedule, loops: Sequence[Var], enabled: bool, threads: int) -> None:
    """Fuse ``loops`` and run them on at most ``threads`` threads of the pool
    (the ``parallel.threads`` annotation)."""
    fused = parallelize(sch, loops, enabled)
    if fused is not None and enabled and sch.extent(fused) > 1 and threads > 0:
        sch.annotate(fused, PARALLEL_THREADS, threads)


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
