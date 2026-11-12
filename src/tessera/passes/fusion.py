"""Operator fusion: partitions a graph into kernels (``FusedGroup``).

A group computes one value. Its structure is fixed so that lowering stays
simple: an optional anchor (one REDUCTION or ANCHOR node), a prologue of
injective and elementwise producers that are inlined into the anchor's
loads, and an epilogue chain of elementwise and broadcast consumers that
runs on the anchor's result, element by element. A group without an anchor
is a tree of elementwise, broadcast and injective nodes.

The partition is built in two sweeps:

1. Forward, every anchor starts a group and grows an epilogue: the anchor's
   result has one reader, it is elementwise or broadcast, and its output has
   the anchor's shape; repeat from that reader.
2. Backward, every node not yet placed fuses into the group of its readers
   when they are all in one group and in positions that can inline it (the
   anchor or a prologue node; any node of an anchor-free group). A cheap
   producer whose readers sit in several such groups is duplicated into each
   (it is recomputed and never stored). Otherwise the node starts a group of
   its own, which later producers may join.

``FusionOptions`` turns each rule off for ablations. ``fuse(enabled=False)``
puts every node in its own group.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from tessera.ir.graph import Graph, Node, Value
from tessera.ir.ops import OpKind, get_op
from tessera.plan import PATTERNS, FusedGroup

# Elementwise and broadcast operators cheap enough to recompute inside the
# loads of a convolution or matrix product, or to duplicate into several
# kernels. Transcendental functions and division are left out.
CHEAP = frozenset(
    {
        "Relu",
        "Neg",
        "Abs",
        "Clip",
        "Cast",
        "Add",
        "Sub",
        "Mul",
        "Maximum",
        "Minimum",
        "Equal",
        "Less",
        "Greater",
        "Where",
    }
)

# C and Metal Shading Language words a kernel may not be named.
_RESERVED = frozenset(
    {
        "auto",
        "break",
        "case",
        "char",
        "const",
        "continue",
        "default",
        "do",
        "double",
        "else",
        "enum",
        "extern",
        "float",
        "for",
        "goto",
        "if",
        "inline",
        "int",
        "long",
        "register",
        "restrict",
        "return",
        "short",
        "signed",
        "sizeof",
        "static",
        "struct",
        "switch",
        "typedef",
        "union",
        "unsigned",
        "void",
        "volatile",
        "while",
        "bool",
        "true",
        "false",
        "main",
        "kernel",
        "device",
        "constant",
        "threadgroup",
        "thread",
        "half",
        "vertex",
        "fragment",
        "uint",
        "uchar",
        "ushort",
        "size_t",
        "ptrdiff_t",
    }
)


@dataclass(frozen=True)
class FusionOptions:
    """Which fusion rules apply (all of them by default)."""

    epilogue: bool = True  # elementwise consumers run after an anchor in its kernel
    prologue: bool = True  # injective and elementwise producers inline into an anchor
    injective: bool = True  # elementwise and injective nodes fuse into one another
    duplicate: bool = True  # cheap producers with several readers are recomputed in each


def kind_of(node: Node) -> OpKind:
    return get_op(node.op).kind


def pattern_of(anchor: Node | None) -> str:
    """The template key (``tessera.plan.PATTERNS``) of a group with this anchor."""
    if anchor is None:
        return "injective"
    match anchor.op:
        case "Conv2d":
            groups = anchor.int_attr("groups")
            nhwc = anchor.str_attr("layout") == "NHWC"
            x, w = anchor.inputs[0].shape, anchor.inputs[1].shape
            channels, out_channels = (x[3], w[3]) if nhwc else (x[1], w[0])
            if groups == 1:
                return "conv2d"
            if groups == channels == out_channels:
                return "depthwise_conv2d"
            raise ValueError(
                f"{anchor.name}: grouped convolution (groups={groups}, {channels} -> "
                f"{out_channels} channels) has no template; passes.optimize splits it"
            )
        case "MatMul":
            return "matmul"
        case "MaxPool2d" | "AvgPool2d":
            return "pool"
        case "ReduceSum" | "ReduceMean" | "ReduceMax":
            return "reduce"
        case "Softmax":
            return "softmax"
        case "LayerNorm":
            return "layernorm"
    raise ValueError(f"{anchor.name}: {anchor.op} cannot anchor a kernel")


@dataclass(eq=False)
class _Group:
    anchor: Node | None
    root: Node  # the node computing the group's output
    members: set[int] = field(default_factory=set)
    prologue: set[int] = field(default_factory=set)
    epilogue: set[int] = field(default_factory=set)


def fuse(
    graph: Graph, *, enabled: bool = True, options: FusionOptions | None = None
) -> list[FusedGroup]:
    """Partition ``graph`` (optimized: canonical operators only) into kernels, in run order."""
    options = options or FusionOptions()
    for node in graph.nodes:
        if kind_of(node) is OpKind.OPAQUE:
            raise ValueError(
                f"{node.op} node {node.name} cannot be fused; run passes.optimize first"
            )
        if len(node.outputs) != 1:
            raise ValueError(f"{node.op} node {node.name} has {len(node.outputs)} outputs")
    if not enabled:
        groups = [_Group(_anchor(node), node, {id(node)}) for node in graph.nodes]
    else:
        groups = _partition(graph, options)
    return _materialize(graph, groups)


def _anchor(node: Node) -> Node | None:
    return node if kind_of(node) in (OpKind.REDUCTION, OpKind.ANCHOR) else None


def _partition(graph: Graph, options: FusionOptions) -> list[_Group]:
    uses = graph.use_map()
    outputs = {id(v) for v in graph.outputs}
    owners: dict[int, list[_Group]] = {}
    groups: list[_Group] = []

    def place(node: Node, group: _Group, role: str = "") -> None:
        group.members.add(id(node))
        if role == "prologue":
            group.prologue.add(id(node))
        elif role == "epilogue":
            group.epilogue.add(id(node))
        owners.setdefault(id(node), []).append(group)

    # Forward sweep: anchors and their epilogues.
    for node in graph.nodes:
        if _anchor(node) is None or id(node) in owners:
            continue
        group = _Group(node, node)
        groups.append(group)
        place(node, group)
        current = node
        while options.epilogue:
            value = current.output
            readers = uses.get(value, [])
            if id(value) in outputs or len(readers) != 1:
                break
            reader = readers[0]
            if (
                id(reader) in owners
                or kind_of(reader) > OpKind.BROADCAST
                or reader.output.shape != node.output.shape
            ):
                break
            place(reader, group, "epilogue")
            group.root = reader
            current = reader

    # Backward sweep: everything else joins its readers' groups or starts one.
    for node in reversed(graph.nodes):
        if id(node) in owners:
            continue
        targets = _join_targets(node, uses, outputs, owners, options)
        if targets is None:
            group = _Group(None, node)
            groups.append(group)
            place(node, group)
        else:
            for group in targets:
                place(node, group, "prologue" if group.anchor is not None else "")
    return groups


def _join_targets(
    node: Node,
    uses: dict[Value, list[Node]],
    outputs: set[int],
    owners: dict[int, list[_Group]],
    options: FusionOptions,
) -> list[_Group] | None:
    """The groups ``node`` fuses into (several if duplicated), or None to start its own."""
    value = node.output
    readers = uses.get(value, [])
    if id(value) in outputs or not readers:
        return None
    targets: list[_Group] = []
    for reader in readers:
        for group in owners[id(reader)]:
            if not _can_inline(node, reader, group, options):
                return None
            if group not in targets:
                targets.append(group)
    if len(targets) > 1 and not (options.duplicate and _cheap(node)):
        return None
    return targets


def _can_inline(producer: Node, reader: Node, group: _Group, options: FusionOptions) -> bool:
    """Whether ``producer`` can be computed inside ``group`` where ``reader`` loads it."""
    if group.anchor is None:
        return options.injective
    if not options.prologue:
        return False
    if reader is not group.anchor and id(reader) not in group.prologue:
        return False  # epilogue operands are loaded, never recomputed
    if kind_of(group.anchor) is OpKind.ANCHOR:
        return _cheap(producer)
    return True


def _cheap(node: Node) -> bool:
    return kind_of(node) is OpKind.INJECTIVE or node.op in CHEAP


def _materialize(graph: Graph, groups: Sequence[_Group]) -> list[FusedGroup]:
    index = {id(n): i for i, n in enumerate(graph.nodes)}
    ordered = sorted(groups, key=lambda g: index[id(g.root)])
    names: set[str] = set()
    out: list[FusedGroup] = []
    for g in ordered:
        nodes = sorted((n for n in graph.nodes if id(n) in g.members), key=lambda n: index[id(n)])
        inside = {id(v) for n in nodes for v in n.outputs}
        inputs = _unique(v for n in nodes for v in n.inputs if id(v) not in inside)
        name = _kernel_name(g.anchor or g.root, names)
        out.append(FusedGroup(name, nodes, inputs, g.root.output, g.anchor, pattern_of(g.anchor)))
    return out


def _unique(values: Iterable[Value]) -> list[Value]:
    seen: set[int] = set()
    out: list[Value] = []
    for v in values:
        if id(v) not in seen:
            seen.add(id(v))
            out.append(v)
    return out


def _kernel_name(node: Node, taken: set[str]) -> str:
    """A C identifier from the node's name, unique among ``taken``."""
    base = node.name.lstrip("_") or node.op.lower()
    if base[0].isdigit():
        base = f"k_{base}"
    if base in _RESERVED:
        base = f"{base}_k"
    name, i = base, 1
    while name in taken:
        name = f"{base}_{i}"
        i += 1
    taken.add(name)
    return name


def verify_groups(graph: Graph, groups: Sequence[FusedGroup]) -> None:
    """Check the ``FusedGroup`` invariants against ``graph``; raise ValueError if broken.

    Every node is in a group; a node is in several groups only if it is a cheap
    producer duplicated into each; groups are in run order; each group's nodes
    are topologically ordered with one output read outside; the prologue
    feeds the anchor, the epilogue has the anchor's shape, and the pattern
    matches the anchor.
    """
    index = {id(n): i for i, n in enumerate(graph.nodes)}
    uses = graph.use_map()
    graph_outputs = {id(v) for v in graph.outputs}
    available = {id(v) for v in graph.inputs}
    membership: dict[int, list[str]] = {}
    roots: set[int] = set()
    names: set[str] = set()

    for g in groups:

        def fail(message: str, g: FusedGroup = g) -> ValueError:
            return ValueError(f"group {g.name}: {message}")

        if not g.name.isidentifier() or g.name in names:
            raise fail("name must be a unique C identifier")
        names.add(g.name)
        if not g.nodes:
            raise fail("has no nodes")
        if any(id(n) not in index for n in g.nodes):
            raise fail("holds a node that is not in the graph")
        positions = [index[id(n)] for n in g.nodes]
        if positions != sorted(set(positions)):
            raise fail("nodes are not in topological order (or repeat)")
        for n in g.nodes:
            membership.setdefault(id(n), []).append(g.name)
        members = {id(n) for n in g.nodes}
        inside = {id(v) for n in g.nodes for v in n.outputs}
        producer = g.output.producer
        if producer is None or id(producer) not in members:
            raise fail(f"output {g.output!r} is not computed by the group")
        roots.add(id(producer))

        expected_inputs = _unique(v for n in g.nodes for v in n.inputs if id(v) not in inside)
        if [id(v) for v in g.inputs] != [id(v) for v in expected_inputs]:
            raise fail("inputs are not the values it reads from outside, in first-use order")
        for v in g.inputs:
            if not (v.is_constant or id(v) in available):
                raise fail(f"reads {v!r} before a previous group computes it")

        heavy = [n for n in g.nodes if kind_of(n) in (OpKind.REDUCTION, OpKind.ANCHOR)]
        if len(heavy) > 1:
            raise fail(f"has {len(heavy)} reduction/anchor nodes")
        if any(kind_of(n) is OpKind.OPAQUE for n in g.nodes):
            raise fail("holds an operator that must be removed by passes.optimize")
        anchor = heavy[0] if heavy else None
        if anchor is not g.anchor:
            raise fail("anchor is not its reduction/anchor node")
        if g.pattern not in PATTERNS or g.pattern != pattern_of(anchor):
            raise fail(f"pattern {g.pattern!r} does not match the anchor")

        if anchor is not None:
            split = g.nodes.index(anchor)
            prologue, epilogue = g.nodes[:split], g.nodes[split + 1 :]
            feeds = {id(anchor)} | {id(n) for n in prologue}
            for n in prologue:
                if kind_of(n) > OpKind.INJECTIVE:
                    raise fail(f"prologue node {n.name} is not injective")
                readers = [r for r in uses.get(n.output, []) if id(r) in members]
                if any(id(r) not in feeds for r in readers):
                    raise fail(f"prologue node {n.name} is read by a node after the anchor")
            for n in epilogue:
                if kind_of(n) > OpKind.BROADCAST:
                    raise fail(f"epilogue node {n.name} is not elementwise or broadcast")
                if n.output.shape != anchor.output.shape:
                    raise fail(f"epilogue node {n.name} changes the anchor's shape")

        for n in g.nodes:
            for v in n.outputs:
                if v is g.output:
                    continue
                if id(v) in graph_outputs:
                    raise fail(f"{v!r} is a graph output but not the group's output")
                for reader in uses.get(v, []):
                    if id(reader) not in members and n not in _groups_with(reader, groups):
                        raise fail(f"{v!r} is read outside the group by {reader.name}")
        available.add(id(g.output))

    for n in graph.nodes:
        owners = membership.get(id(n))
        if not owners:
            raise ValueError(f"node {n.name} is in no group")
        if len(owners) > 1 and (id(n) in roots or not _cheap(n)):
            raise ValueError(f"node {n.name} is in groups {owners} but cannot be duplicated")
    for v in graph.outputs:
        if not (v.is_constant or id(v) in available):
            raise ValueError(f"graph output {v!r} is computed by no group")


def _groups_with(node: Node, groups: Sequence[FusedGroup]) -> list[Node]:
    """Every node sharing a group with ``node`` (to accept reads of duplicated producers)."""
    out: list[Node] = []
    for g in groups:
        if any(n is node for n in g.nodes):
            out.extend(g.nodes)
    return out
