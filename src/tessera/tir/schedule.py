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
from tessera.tir.stmt import PrimFunc, Scope


class ScheduleError(Exception):
    """A primitive cannot be applied (it would change the result or the IR is not
    in the form it needs)."""


class Schedule:
    def __init__(self, func: PrimFunc) -> None:
        self._func = func
        self.trace: list[str] = []  # one line per applied primitive, for `tessera inspect`

    @property
    def func(self) -> PrimFunc:
        return self._func

    def copy(self) -> Schedule:
        raise NotImplementedError

    # -- lookup -------------------------------------------------------------

    def blocks(self) -> list[str]:
        """Block names in program order (the root block excluded)."""
        raise NotImplementedError

    def get_loops(self, block: str) -> list[Var]:
        """The loops around ``block``, outermost first."""
        raise NotImplementedError

    def extent(self, loop: Var) -> int:
        """The constant trip count of ``loop``."""
        raise NotImplementedError

    # -- loops --------------------------------------------------------------

    def split(self, loop: Var, factors: Sequence[int | None]) -> list[Var]:
        """Split into ``len(factors)`` nested loops, outermost first. At most one
        factor may be None (inferred). If the product exceeds the extent, the
        block gets a predicate (imperfect split)."""
        raise NotImplementedError

    def fuse(self, *loops: Var) -> Var:
        """Fuse perfectly nested consecutive loops into one."""
        raise NotImplementedError

    def reorder(self, *loops: Var) -> None:
        """Reorder loops of one perfect nest into the given order."""
        raise NotImplementedError

    def parallel(self, loop: Var) -> None:
        raise NotImplementedError

    def vectorize(self, loop: Var) -> None:
        raise NotImplementedError

    def unroll(self, loop: Var) -> None:
        raise NotImplementedError

    def bind(self, loop: Var, thread_tag: str) -> None:
        """Bind to a Metal grid axis, one of ``tessera.tir.stmt.THREAD_TAGS``."""
        raise NotImplementedError

    def annotate(self, target: Var | str, key: str, value: int | float | str) -> None:
        """Attach a backend hint to a loop (Var) or a block (name)."""
        raise NotImplementedError

    # -- blocks -------------------------------------------------------------

    def compute_inline(self, block: str) -> None:
        """Inline an elementwise/injective producer into all its consumers."""
        raise NotImplementedError

    def reverse_compute_inline(self, block: str) -> None:
        """Inline a consumer (epilogue) into its only producer."""
        raise NotImplementedError

    def compute_at(self, block: str, loop: Var) -> None:
        """Move a producer under ``loop`` of its consumer, computing only the
        region the consumer needs there."""
        raise NotImplementedError

    def reverse_compute_at(self, block: str, loop: Var) -> None:
        """Move a consumer under ``loop`` of its producer (fuse an epilogue into tiles)."""
        raise NotImplementedError

    def cache_read(self, block: str, read_index: int, scope: Scope) -> str:
        """Copy the buffer ``block`` reads at ``read_index`` into a new buffer of
        ``scope``; returns the name of the copy block (place it with compute_at)."""
        raise NotImplementedError

    def cache_write(self, block: str, write_index: int, scope: Scope) -> str:
        """Make ``block`` write into a new buffer of ``scope`` and copy it out;
        returns the name of the copy-out block."""
        raise NotImplementedError

    def decompose_reduction(self, block: str, loop: Var) -> str:
        """Split a reduction's init into its own block above ``loop``; returns its name."""
        raise NotImplementedError

    def tensorize(self, loop: Var, intrinsic: str) -> None:
        """Replace the nest rooted at ``loop`` with a registered intrinsic
        (see ``tessera.tir.intrin``), after checking the nest computes what
        the intrinsic's description computes."""
        raise NotImplementedError
