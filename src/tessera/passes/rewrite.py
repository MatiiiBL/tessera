"""Local graph rewriting: rules that replace a node's output with an equivalent value.

A rule looks at one node and either returns a value computing the same thing
(often built from new nodes appended to the graph) or None. ``Rewriter``
applies a list of rules to every node in topological order, sweeping until
no rule fires. Replacements are recorded and applied to the readers when
they are visited, so a sweep is linear in the size of the graph; at the end
of each sweep the graph is re-sorted and dead nodes are removed.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence

import numpy as np

from tessera.ir.graph import Attr, Graph, NDArray, Node, Value

type Rule = Callable[[Rewriter, Node], Value | None]


class Rewriter:
    """Applies rules to a graph in place."""

    def __init__(self, graph: Graph, *, fold: bool = True) -> None:
        self.graph = graph
        self.fold = fold  # whether rules may compute on constants at compile time
        self._subst: dict[int, Value] = {}
        self.uses: dict[Value, list[Node]] = {}

    # -- building ---------------------------------------------------------------

    def op(self, op: str, *inputs: Value, name: str | None = None, **attrs: Attr) -> Value:
        """Append a single-output node (it is sorted into place at the end of the sweep)."""
        return self.graph.op(op, *inputs, name=name, **attrs)

    def const(
        self, data: NDArray | float | int, name: str = "const", like: Value | None = None
    ) -> Value:
        """A new constant; ``like`` gives its element type."""
        array = np.asarray(data, like.dtype.numpy if like is not None else None)
        return self.graph.add_constant(array, name)

    def readers(self, value: Value) -> list[Node]:
        """Nodes reading ``value`` when the sweep started."""
        return self.uses.get(value, [])

    def single_reader(self, value: Value) -> bool:
        """``value`` has one reader and is not a graph output."""
        return len(self.readers(value)) == 1 and all(v is not value for v in self.graph.outputs)

    # -- running ----------------------------------------------------------------

    def replace(self, old: Value, new: Value) -> None:
        if old.type != new.type:
            raise ValueError(f"cannot replace {old!r} with {new!r}: types differ")
        if old is not new:
            self._subst[id(old)] = new

    def resolve(self, value: Value) -> Value:
        while id(value) in self._subst:
            value = self._subst[id(value)]
        return value

    def run(self, rules: Sequence[Rule], *, max_sweeps: int = 100) -> bool:
        """Sweep until no rule fires; return whether anything changed."""
        changed = False
        for _ in range(max_sweeps):
            fired = False
            self.uses = self.graph.use_map()
            for node in list(self.graph.nodes):
                node.inputs = [self.resolve(v) for v in node.inputs]
                # A replacement may take the node's exact name (the node dies with the sweep).
                names = {node.name, *(v.name for v in node.outputs)}
                for name in names:
                    self.graph.release_name(name)
                for rule in rules:
                    new = rule(self, node)
                    if new is not None:
                        self.replace(node.output, new)
                        fired = True
                        break
                for name in names:
                    self._reclaim(name)
            self.finish_sweep()
            changed |= fired
            if not fired:
                return changed
        raise RuntimeError(f"rewriting did not converge after {max_sweeps} sweeps")

    def _reclaim(self, name: str) -> None:
        with contextlib.suppress(ValueError):  # a replacement took the name
            self.graph.claim_name(name)

    def finish_sweep(self) -> None:
        g = self.graph
        for node in g.nodes:
            node.inputs = [self.resolve(v) for v in node.inputs]
        outputs = [self.resolve(v) for v in g.outputs]
        g.set_outputs(outputs)
        self._subst.clear()
        g.topo_sort()
        eliminate_dead_code(g)


def eliminate_dead_code(graph: Graph) -> None:
    """Remove nodes whose outputs no graph output depends on."""
    live: set[int] = {id(v) for v in graph.outputs}
    kept: list[Node] = []
    for node in reversed(graph.nodes):
        if any(id(v) in live for v in node.outputs):
            kept.append(node)
            live.update(id(v) for v in node.inputs)
    kept.reverse()
    graph.nodes = kept


def constant(value: Value) -> NDArray | None:
    """The data of a constant value, else None."""
    return value.data


def scalar(value: Value) -> float | None:
    """The value of a constant whose elements are all equal, else None."""
    data = value.data
    if data is None or data.size == 0 or data.dtype == np.bool_:
        return None
    first = data.reshape(-1)[0]
    if not np.all(data == first):
        return None
    return float(first)
