"""Memory planning: where every value a kernel reads or writes lives at run time.

Graph inputs and outputs are passed by the caller (INPUT, OUTPUT). Constants
are packed into one aligned blob (CONSTANT). Every other value read across
groups gets an offset in a scratch arena (ARENA): values are placed in group
order by greedy best fit, and a value's bytes are reused once its last reader
has run.

A view group (only Reshape nodes) runs no kernel: its output shares the
storage of its input. When a view's output is a graph output and its input
is computed by a kernel, that kernel writes into the output buffer directly.
A graph output that is a graph input, a constant, or the view of one keeps
that storage; the caller copies it out.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tessera.ir.graph import Graph, NDArray, Value
from tessera.plan import BufferRef, FusedGroup, MemoryPlan, StorageKind


def round_up(n: int, alignment: int) -> int:
    return -(-n // alignment) * alignment


@dataclass
class _Storage:
    """A set of values sharing bytes (a value and its views)."""

    root: Value
    first: int  # index of the group that writes it
    last: int  # index of the last group that reads it (or a view of it)
    ref: BufferRef | None = None


def plan_memory(graph: Graph, groups: Sequence[FusedGroup], *, alignment: int = 256) -> MemoryPlan:
    """Assign a ``BufferRef`` to every value crossing a group boundary."""
    if alignment < 1 or alignment & (alignment - 1):
        raise ValueError(f"alignment must be a power of two, got {alignment}")
    refs: dict[str, BufferRef] = {}
    for i, v in enumerate(graph.inputs):
        refs[v.name] = BufferRef(StorageKind.INPUT, 0, v.type.nbytes, i)

    constants = _constants(graph, groups)
    blob, offsets = _pack(constants, alignment)
    for v in constants:
        refs.setdefault(v.name, BufferRef(StorageKind.CONSTANT, offsets[id(v)], v.type.nbytes))
    for i, v in enumerate(graph.outputs):
        refs.setdefault(v.name, BufferRef(StorageKind.OUTPUT, 0, v.type.nbytes, i))

    # Storage classes: each view output joins the class of its input.
    storage: dict[int, _Storage] = {}
    for pos, g in enumerate(groups):
        for v in g.inputs:
            s = storage.get(id(v))
            if s is not None:
                s.last = max(s.last, pos)
        if g.is_view and len(g.inputs) == 1 and id(g.inputs[0]) in storage:
            s = storage[id(g.inputs[0])]
            storage[id(g.output)] = s
        elif g.is_view and len(g.inputs) == 1:
            continue  # a view of an input or a constant: aliased below
        else:
            storage[id(g.output)] = _Storage(g.output, pos, pos)

    # A view that is a graph output lets its kernel write into the output buffer.
    for v in graph.outputs:
        s = storage.get(id(v))
        if s is not None and s.ref is None and s.root.name not in refs:
            s.ref = refs[v.name]
    for s in {id(s): s for s in storage.values()}.values():
        if s.ref is None and s.root.name in refs:
            s.ref = refs[s.root.name]

    arena = _Arena(alignment)
    classes = sorted({id(s): s for s in storage.values()}.values(), key=lambda s: s.first)
    by_last: dict[int, list[_Storage]] = {}
    for s in classes:
        by_last.setdefault(s.last, []).append(s)
    pending = iter(classes)
    upcoming = next(pending, None)
    for pos in range(len(groups)):
        while upcoming is not None and upcoming.first == pos:
            if upcoming.ref is None:
                nbytes = upcoming.root.type.nbytes
                upcoming.ref = BufferRef(StorageKind.ARENA, arena.allocate(nbytes), nbytes)
            upcoming = next(pending, None)
        for s in by_last.get(pos, []):
            if s.ref is not None and s.ref.kind is StorageKind.ARENA:
                arena.release(s.ref.offset, s.ref.nbytes)

    for pos, g in enumerate(groups):
        s = storage.get(id(g.output))
        if s is not None:
            assert s.ref is not None
            refs.setdefault(g.output.name, s.ref)
        elif g.is_view:
            refs.setdefault(g.output.name, refs[g.inputs[0].name])
        for v in g.inputs:
            if v.name not in refs:
                raise ValueError(
                    f"group {g.name} (position {pos}) reads {v!r}, which has no storage"
                )
    return MemoryPlan(refs, arena.size, blob, alignment)


def _constants(graph: Graph, groups: Sequence[FusedGroup]) -> list[Value]:
    """Constants read by kernels or returned as outputs, in first-use order."""
    seen: set[int] = set()
    out: list[Value] = []
    for v in [*(v for g in groups for v in g.inputs), *graph.outputs]:
        if v.is_constant and id(v) not in seen:
            seen.add(id(v))
            out.append(v)
    return out


def _pack(constants: Sequence[Value], alignment: int) -> tuple[NDArray, dict[int, int]]:
    offsets: dict[int, int] = {}
    size = 0
    for v in constants:
        offsets[id(v)] = size
        size = round_up(size + v.type.nbytes, alignment)
    blob = np.zeros(size, np.uint8)
    for v in constants:
        assert v.data is not None
        raw = np.ascontiguousarray(v.data).reshape(-1).view(np.uint8)
        blob[offsets[id(v)] : offsets[id(v)] + raw.size] = raw
    return blob, offsets


class _Arena:
    """Best-fit allocation over a free list of aligned blocks."""

    def __init__(self, alignment: int) -> None:
        self.alignment = alignment
        self.size = 0
        self.free: list[tuple[int, int]] = []  # (offset, size), sorted by offset

    def allocate(self, nbytes: int) -> int:
        size = round_up(nbytes, self.alignment)
        if size == 0:
            return 0
        best = None
        for i, (_, block) in enumerate(self.free):
            if block >= size and (best is None or block < self.free[best][1]):
                best = i
        if best is not None:
            offset, block = self.free[best]
            if block == size:
                del self.free[best]
            else:
                self.free[best] = (offset + size, block - size)
            return offset
        # No block fits: grow the arena, reusing a free block that ends at the top.
        touches_top = bool(self.free) and sum(self.free[-1]) == self.size
        offset = self.free.pop()[0] if touches_top else self.size
        self.size = offset + size
        return offset

    def release(self, offset: int, nbytes: int) -> None:
        size = round_up(nbytes, self.alignment)
        if size == 0:
            return
        blocks = sorted([*self.free, (offset, size)])
        merged: list[tuple[int, int]] = []
        for start, length in blocks:
            if merged and sum(merged[-1]) == start:
                merged[-1] = (merged[-1][0], merged[-1][1] + length)
            else:
                merged.append((start, length))
        self.free = merged
