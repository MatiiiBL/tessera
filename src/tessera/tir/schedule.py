"""Schedule primitives: semantics-preserving rewrites of a PrimFunc's loops.

This module fixes the API that backend templates program against. Loops are
named by their loop variable (``Var``, identity) and blocks by name. Every
primitive returns new handles where it creates loops or blocks; handles to
loops it removed are invalid afterwards. ``Schedule.func`` is the current
function. A primitive that cannot be applied safely raises ``ScheduleError``
and leaves the schedule unchanged.

Implemented by the tir package (docs/tir.md); the signatures below are the
contract.
"""

from __future__ import annotations

from collections.abc import Sequence

from tessera.tir.expr import Var
from tessera.tir.primitives import blocks as _blocks
from tessera.tir.primitives import cache as _cache
from tessera.tir.primitives import loops as _loops
from tessera.tir.primitives.common import ScheduleError, block_paths, find_loop, loop_of, realize_of
from tessera.tir.stmt import ForKind, PrimFunc, Scope

__all__ = ["Schedule", "ScheduleError"]


class Schedule:
    def __init__(self, func: PrimFunc) -> None:
        self._func = func
        self.trace: list[str] = []  # one line per applied primitive, for `tessera inspect`

    @property
    def func(self) -> PrimFunc:
        return self._func

    def _apply(self, func: PrimFunc, line: str) -> None:
        self._func = func
        self.trace.append(line)

    def copy(self) -> Schedule:
        out = Schedule(self._func)
        out.trace = list(self.trace)
        return out

    # -- lookup -------------------------------------------------------------

    def blocks(self) -> list[str]:
        """Block names in program order (the root block excluded)."""
        return [realize_of(p).block.name for p in block_paths(self._func)]

    def get_loops(self, block: str) -> list[Var]:
        """The loops around ``block``, outermost first."""
        return _loops.get_loops(self._func, block)

    def extent(self, loop: Var) -> int:
        """The constant trip count of ``loop``."""
        return _loops.const_extent(loop_of(find_loop(self._func, loop)))

    # -- loops --------------------------------------------------------------

    def split(self, loop: Var, factors: Sequence[int | None]) -> list[Var]:
        """Split into ``len(factors)`` nested loops, outermost first. At most one
        factor may be None (inferred). If the product exceeds the extent, the
        block gets a predicate (imperfect split)."""
        func, new = _loops.split(self._func, loop, factors)
        self._apply(func, f"split({loop.name}, {list(factors)}) -> {_names(new)}")
        return new

    def fuse(self, *loops: Var) -> Var:
        """Fuse perfectly nested consecutive loops into one."""
        func, new = _loops.fuse(self._func, loops)
        self._apply(func, f"fuse({_names(loops)}) -> {new.name}")
        return new

    def reorder(self, *loops: Var) -> None:
        """Reorder loops of one perfect nest into the given order."""
        self._apply(_loops.reorder(self._func, loops), f"reorder({_names(loops)})")

    def parallel(self, loop: Var) -> None:
        self._apply(_loops.set_kind(self._func, loop, ForKind.PARALLEL), f"parallel({loop.name})")

    def vectorize(self, loop: Var) -> None:
        func = _loops.set_kind(self._func, loop, ForKind.VECTORIZED)
        self._apply(func, f"vectorize({loop.name})")

    def unroll(self, loop: Var) -> None:
        self._apply(_loops.set_kind(self._func, loop, ForKind.UNROLLED), f"unroll({loop.name})")

    def bind(self, loop: Var, thread_tag: str) -> None:
        """Bind to a Metal grid axis, one of ``tessera.tir.stmt.THREAD_TAGS``."""
        func = _loops.set_kind(self._func, loop, ForKind.THREAD_BINDING, thread_tag)
        self._apply(func, f"bind({loop.name}, {thread_tag!r})")

    def annotate(self, target: Var | str, key: str, value: int | float | str) -> None:
        """Attach a backend hint to a loop (Var) or a block (name)."""
        if isinstance(target, Var):
            func = _loops.annotate_loop(self._func, target, key, value)
            label = target.name
        else:
            func = _blocks.annotate_block(self._func, target, key, value)
            label = repr(target)
        self._apply(func, f"annotate({label}, {key!r}, {value!r})")

    # -- blocks -------------------------------------------------------------

    def compute_inline(self, block: str) -> None:
        """Inline an elementwise/injective producer into all its consumers."""
        self._apply(_blocks.compute_inline(self._func, block), f"compute_inline({block!r})")

    def reverse_compute_inline(self, block: str) -> None:
        """Inline a consumer (epilogue) into its only producer."""
        func = _blocks.reverse_compute_inline(self._func, block)
        self._apply(func, f"reverse_compute_inline({block!r})")

    def compute_at(self, block: str, loop: Var) -> None:
        """Move a producer under ``loop`` of its consumer, computing only the
        region the consumer needs there."""
        func = _blocks.compute_at(self._func, block, loop)
        self._apply(func, f"compute_at({block!r}, {loop.name})")

    def reverse_compute_at(self, block: str, loop: Var) -> None:
        """Move a consumer under ``loop`` of its producer (fuse an epilogue into tiles)."""
        func = _blocks.reverse_compute_at(self._func, block, loop)
        self._apply(func, f"reverse_compute_at({block!r}, {loop.name})")

    def cache_read(self, block: str, read_index: int, scope: Scope) -> str:
        """Copy the buffer ``block`` reads at ``read_index`` into a new buffer of
        ``scope``; returns the name of the copy block (place it with compute_at)."""
        func, name = _cache.cache_read(self._func, block, read_index, scope)
        self._apply(func, f"cache_read({block!r}, {read_index}, {scope!r}) -> {name!r}")
        return name

    def cache_write(self, block: str, write_index: int, scope: Scope) -> str:
        """Make ``block`` write into a new buffer of ``scope`` and copy it out;
        returns the name of the copy-out block."""
        func, name = _cache.cache_write(self._func, block, write_index, scope)
        self._apply(func, f"cache_write({block!r}, {write_index}, {scope!r}) -> {name!r}")
        return name

    def decompose_reduction(self, block: str, loop: Var) -> str:
        """Split a reduction's init into its own block above ``loop``; returns its name."""
        func, name = _cache.decompose_reduction(self._func, block, loop)
        self._apply(func, f"decompose_reduction({block!r}, {loop.name}) -> {name!r}")
        return name

    def tensorize(self, loop: Var, intrinsic: str) -> None:
        """Replace the nest rooted at ``loop`` with a registered intrinsic
        (see ``tessera.tir.intrin``), after checking the nest computes what
        the intrinsic's description computes."""
        raise NotImplementedError


def _names(loops: Sequence[Var]) -> str:
    return "[" + ", ".join(v.name for v in loops) + "]"
