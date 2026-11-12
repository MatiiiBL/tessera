"""Constant folding and common-subexpression elimination."""

from __future__ import annotations

import hashlib

import numpy as np

from tessera.ir.graph import Attr, Graph, Node, Value
from tessera.passes.rewrite import Rewriter, Rule

# Folding is skipped when it would turn small constants into a much larger one
# (an Expand of a scalar, for example): the kernel computing it is cheaper than
# the bytes it would add to the constant blob.
GROWTH_LIMIT = 4
GROWTH_MIN_BYTES = 1 << 20


def _fold(rw: Rewriter, node: Node) -> Value | None:
    if not node.inputs or not all(v.data is not None for v in node.inputs):
        return None
    in_bytes = sum(v.type.nbytes for v in node.inputs)
    out_bytes = node.output.type.nbytes
    if out_bytes > GROWTH_MIN_BYTES and out_bytes > GROWTH_LIMIT * in_bytes:
        return None
    args = [v.data for v in node.inputs if v.data is not None]
    with np.errstate(all="ignore"):
        (result,) = node.opdef.reference(args, node.attrs)
    data = np.asarray(np.asarray(result).astype(node.output.dtype.numpy, copy=False), order="C")
    return rw.const(data, node.name)


FOLD: list[Rule] = [_fold]


def _attr_key(attrs: dict[str, Attr]) -> tuple[tuple[str, Attr], ...]:
    return tuple(sorted(attrs.items()))


def merge_constants(graph: Graph) -> bool:
    """Make equal constants (same type, shape and bytes) one value; return whether any merged.

    Constants are first bucketed by type, shape and their first elements, so
    only the few that could be equal are hashed in full (weights rarely are).
    """
    buckets: dict[tuple[str, tuple[int, ...], bytes], list[Value]] = {}
    for v in graph.constants():
        assert v.data is not None
        head = np.asarray(v.data, order="C").reshape(-1)[:16].tobytes()
        buckets.setdefault((str(v.dtype), v.shape, head), []).append(v)
    subst: dict[int, Value] = {}
    for candidates in buckets.values():
        if len(candidates) < 2:
            continue
        canonical: dict[bytes, Value] = {}
        for v in candidates:
            assert v.data is not None
            data = np.asarray(v.data, order="C").tobytes()
            first = canonical.setdefault(hashlib.blake2b(data, digest_size=16).digest(), v)
            if first is not v:
                subst[id(v)] = first
    if not subst:
        return False
    for node in graph.nodes:
        node.inputs = [subst.get(id(v), v) for v in node.inputs]
    graph.set_outputs([subst.get(id(v), v) for v in graph.outputs])
    return True


def eliminate_common_subexpressions(graph: Graph) -> bool:
    """Merge nodes with the same operator, inputs and attributes."""
    merged = merge_constants(graph)
    rw = Rewriter(graph)
    seen: dict[tuple[object, ...], Node] = {}
    fired = False
    for node in graph.nodes:
        node.inputs = [rw.resolve(v) for v in node.inputs]
        key = (node.op, tuple(id(v) for v in node.inputs), _attr_key(node.attrs))
        first = seen.setdefault(key, node)
        if first is not node:
            for old, new in zip(node.outputs, first.outputs, strict=True):
                rw.replace(old, new)
            fired = True
    rw.finish_sweep()
    return merged or fired
