"""The optimization pipeline: an imported graph to canonical, folded form."""

from __future__ import annotations

from tessera.ir.graph import Graph, Value
from tessera.ir.opdefs import IMPORT_OPS
from tessera.passes.canonicalize import CANONICALIZE, SIMPLIFY, TRANSPOSE_AS_RESHAPE
from tessera.passes.fold import FOLD, eliminate_common_subexpressions
from tessera.passes.rewrite import Rewriter, eliminate_dead_code

LAYOUTS = ("NHWC", "NCHW")


def optimize(graph: Graph, *, layout: str = "NHWC", fold: bool = True) -> Graph:
    """A new graph computing the same outputs with canonical operators only.

    ONNX-only operators are rewritten, no-op nodes removed, nodes on
    constants evaluated at compile time (unless ``fold`` is False), equal
    subexpressions merged and dead nodes removed. Input and output names are
    kept.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    g = graph.copy()
    output_names = [v.name for v in g.outputs]
    eliminate_dead_code(g)

    rw = Rewriter(g, fold=fold)
    rw.run(CANONICALIZE + SIMPLIFY + TRANSPOSE_AS_RESHAPE)
    _cleanup(rw)

    leftover = sorted({n.op for n in g.nodes if n.op in IMPORT_OPS})
    if leftover:
        raise ValueError(f"operators {leftover} could not be canonicalized")
    _restore_output_names(g, output_names)
    g.verify()
    return g


def _cleanup(rw: Rewriter) -> None:
    """Simplify, fold and merge until nothing changes."""
    rules = SIMPLIFY + TRANSPOSE_AS_RESHAPE + (FOLD if rw.fold else [])
    while True:
        changed = rw.run(rules)
        changed |= eliminate_common_subexpressions(rw.graph)
        if not changed:
            return


def _restore_output_names(g: Graph, names: list[str]) -> None:
    """Give every output back its name.

    An output that became a graph input, a constant or the same value as an
    earlier output cannot be renamed; it is returned through a Reshape to its
    own shape (a view, which the memory plan turns into a copy).
    """
    live = {v.name: v for v in g.values()}
    outputs: list[Value] = []
    for value, name in zip(g.outputs, names, strict=True):
        if value.name == name and value not in outputs:
            outputs.append(value)
            continue
        owner = live.get(name)
        renamable = (
            value.producer is not None
            and value not in outputs
            and value.name not in names
            and (owner is None or owner is value)
        )
        if renamable:
            del live[value.name]
            value.name = name
            assert value.producer is not None
            if len(value.producer.outputs) == 1:
                value.producer.name = name
        else:
            if owner is not None and owner is not value:
                owner.name = g.fresh_name(name)
                if owner.producer is not None and len(owner.producer.outputs) == 1:
                    owner.producer.name = owner.name
            value = g.op("Reshape", value, shape=value.shape)
            value.name = name
            assert value.producer is not None
            value.producer.name = name
        live[name] = value
        outputs.append(value)
    g.set_outputs(outputs)
