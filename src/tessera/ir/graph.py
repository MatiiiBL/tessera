"""The graph IR: a dataflow graph of tensor operators with static shapes.

A ``Graph`` is a list of ``Node``s in topological order. Each node reads
``Value``s and defines new ones (SSA: every value has exactly one definition,
which is a graph input, a constant or a node output). Shapes are static:
a model is compiled for one input shape.

Values compare by identity. Names are unique within a graph and are used for
printing, for kernel names and as keys in the memory plan.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from tessera.dtype import DType

if TYPE_CHECKING:
    from tessera.ir.ops import OpDef

type Attr = int | float | str | tuple[int, ...] | tuple[float, ...] | tuple[str, ...]
type NDArray = npt.NDArray[Any]


@dataclass(frozen=True, slots=True)
class TensorType:
    """A static shape and an element type."""

    shape: tuple[int, ...]
    dtype: DType

    def __post_init__(self) -> None:
        if any(not isinstance(d, int) or d < 0 for d in self.shape):
            raise ValueError(f"shape must be non-negative ints, got {self.shape}")

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        return prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.numel * self.dtype.itemsize

    @classmethod
    def of(cls, array: NDArray) -> TensorType:
        return cls(tuple(int(d) for d in array.shape), DType.from_numpy(array.dtype))

    def __str__(self) -> str:
        return f"{self.dtype}[{','.join(map(str, self.shape))}]"


class Value:
    """An SSA value: a graph input, a constant (``data`` is set) or a node output."""

    __slots__ = ("data", "name", "producer", "type")

    def __init__(
        self,
        name: str,
        type: TensorType,
        producer: Node | None = None,
        data: NDArray | None = None,
    ) -> None:
        self.name = name
        self.type = type
        self.producer = producer
        self.data = data

    @property
    def is_constant(self) -> bool:
        return self.data is not None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.type.shape

    @property
    def dtype(self) -> DType:
        return self.type.dtype

    def __repr__(self) -> str:
        return f"%{self.name}: {self.type}"


class Node:
    """One operator application."""

    __slots__ = ("attrs", "inputs", "name", "op", "outputs")

    def __init__(
        self,
        op: str,
        inputs: Sequence[Value],
        outputs: Sequence[Value],
        attrs: Mapping[str, Attr] | None = None,
        name: str = "",
    ) -> None:
        self.op = op
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.attrs: dict[str, Attr] = dict(attrs or {})
        self.name = name or op.lower()
        for out in self.outputs:
            out.producer = self

    @property
    def output(self) -> Value:
        """The only output (most operators have one)."""
        if len(self.outputs) != 1:
            raise ValueError(f"{self.op} node {self.name} has {len(self.outputs)} outputs")
        return self.outputs[0]

    @property
    def opdef(self) -> OpDef:
        from tessera.ir.ops import get_op

        return get_op(self.op)

    def int_attr(self, key: str, default: int | None = None) -> int:
        value = self.attrs.get(key, default)
        if not isinstance(value, int):
            raise TypeError(f"{self.op}.{key} must be an int, got {value!r}")
        return value

    def float_attr(self, key: str, default: float | None = None) -> float:
        value = self.attrs.get(key, default)
        if not isinstance(value, int | float):
            raise TypeError(f"{self.op}.{key} must be a number, got {value!r}")
        return float(value)

    def str_attr(self, key: str, default: str | None = None) -> str:
        value = self.attrs.get(key, default)
        if not isinstance(value, str):
            raise TypeError(f"{self.op}.{key} must be a string, got {value!r}")
        return value

    def ints_attr(self, key: str, default: tuple[int, ...] | None = None) -> tuple[int, ...]:
        value = self.attrs.get(key, default)
        if not isinstance(value, tuple) or not all(isinstance(v, int) for v in value):
            raise TypeError(f"{self.op}.{key} must be a tuple of ints, got {value!r}")
        return tuple(int(v) for v in value)

    def __repr__(self) -> str:
        ins = ", ".join(f"%{v.name}" for v in self.inputs)
        outs = ", ".join(f"%{v.name}" for v in self.outputs)
        attrs = "".join(f" {k}={v!r}" for k, v in sorted(self.attrs.items()))
        return f"{outs} = {self.op}({ins}){attrs}"


_NAME_SAFE = re.compile(r"[^A-Za-z0-9_]")


class Graph:
    """A model: inputs, constants, nodes in topological order and outputs."""

    def __init__(self, name: str = "graph") -> None:
        self.name = name
        self.inputs: list[Value] = []
        self.outputs: list[Value] = []
        self.nodes: list[Node] = []
        self._names: set[str] = set()

    # -- construction -------------------------------------------------------

    def fresh_name(self, hint: str) -> str:
        """A name not used yet in this graph, made only of [A-Za-z0-9_]."""
        base = _NAME_SAFE.sub("_", hint) or "v"
        if base[0].isdigit():
            base = f"v{base}"
        name, i = base, 1
        while name in self._names:
            name = f"{base}_{i}"
            i += 1
        self._names.add(name)
        return name

    def add_input(self, name: str, type: TensorType) -> Value:
        value = Value(self.fresh_name(name), type)
        self.inputs.append(value)
        return value

    def add_constant(self, data: npt.ArrayLike, name: str = "const") -> Value:
        array = np.ascontiguousarray(data)
        return Value(self.fresh_name(name), TensorType.of(array), data=array)

    def add_node(
        self,
        op: str,
        inputs: Sequence[Value],
        attrs: Mapping[str, Attr] | None = None,
        *,
        name: str | None = None,
        output_types: Sequence[TensorType] | None = None,
    ) -> Node:
        """Append a node; output types come from the op's shape inference unless given.

        Attributes are normalized first (defaults filled in, negative axes made
        non-negative), so ``attrs`` may use the lenient ONNX-like spelling.
        """
        from tessera.ir.ops import ShapeError, get_op

        opdef = get_op(op)
        types = [v.type for v in inputs]
        try:
            attrs = opdef.normalize(types, attrs or {}) if opdef.normalize else dict(attrs or {})
            if output_types is None:
                output_types = opdef.infer(types, attrs)
        except ShapeError as e:
            raise ShapeError(f"{op} node {name or op.lower()!r}: {e}") from None
        node_name = self.fresh_name(name or op.lower())
        outs = [
            Value(node_name if len(output_types) == 1 else self.fresh_name(f"{node_name}_{i}"), t)
            for i, t in enumerate(output_types)
        ]
        node = Node(op, inputs, outs, attrs, node_name)
        self.nodes.append(node)
        return node

    def op(
        self,
        op: str,
        *inputs: Value,
        name: str | None = None,
        **attrs: Attr,
    ) -> Value:
        """Append a single-output node and return its output."""
        return self.add_node(op, inputs, attrs, name=name).output

    def set_outputs(self, values: Iterable[Value]) -> None:
        self.outputs = list(values)

    def claim_name(self, name: str) -> None:
        """Reserve a name created elsewhere (frontends that build Values directly)."""
        if name in self._names:
            raise ValueError(f"name {name!r} is already used")
        self._names.add(name)

    # -- queries ------------------------------------------------------------

    def values(self) -> Iterator[Value]:
        """Every value defined in the graph: inputs, constants read by nodes, node outputs."""
        seen: set[int] = set()
        for v in self.inputs:
            seen.add(id(v))
            yield v
        for node in self.nodes:
            for v in node.inputs:
                if id(v) not in seen:
                    seen.add(id(v))
                    yield v
            for v in node.outputs:
                if id(v) not in seen:
                    seen.add(id(v))
                    yield v
        for v in self.outputs:
            if id(v) not in seen:
                seen.add(id(v))
                yield v

    def constants(self) -> list[Value]:
        return [v for v in self.values() if v.is_constant]

    def use_map(self) -> dict[Value, list[Node]]:
        """Every value to the nodes that read it (graph outputs are not nodes)."""
        uses: dict[Value, list[Node]] = {}
        for node in self.nodes:
            for v in node.inputs:
                users = uses.setdefault(v, [])
                if node not in users:
                    users.append(node)
        return uses

    # -- rewriting ----------------------------------------------------------

    def replace_all_uses(self, old: Value, new: Value) -> None:
        if old.type != new.type:
            raise ValueError(f"cannot replace {old!r} with {new!r}: types differ")
        for node in self.nodes:
            node.inputs = [new if v is old else v for v in node.inputs]
        self.outputs = [new if v is old else v for v in self.outputs]

    def remove_nodes(self, nodes: Iterable[Node]) -> None:
        dead = {id(n) for n in nodes}
        self.nodes = [n for n in self.nodes if id(n) not in dead]

    def topo_sort(self) -> None:
        """Reorder nodes so that every value is defined before it is read.

        Nodes keep their order where it is already valid; a node that comes
        too late moves to just before its first reader.
        """
        members = {id(n) for n in self.nodes}
        inputs = {id(v) for v in self.inputs}
        placed: set[int] = set()
        on_stack: set[int] = set()
        ordered: list[Node] = []
        for root in self.nodes:
            if id(root) in placed:
                continue
            stack: list[tuple[Node, int]] = [(root, 0)]
            on_stack.add(id(root))
            while stack:
                node, i = stack[-1]
                if i == len(node.inputs):
                    stack.pop()
                    on_stack.discard(id(node))
                    placed.add(id(node))
                    ordered.append(node)
                    continue
                stack[-1] = (node, i + 1)
                v = node.inputs[i]
                if v.is_constant or id(v) in inputs:
                    continue
                producer = v.producer
                if producer is None or id(producer) not in members:
                    raise ValueError(f"{node!r} reads {v!r}, which no node defines")
                if id(producer) in placed:
                    continue
                if id(producer) in on_stack:
                    raise ValueError(f"graph has a cycle through {producer!r}")
                on_stack.add(id(producer))
                stack.append((producer, 0))
        self.nodes = ordered

    def verify(self) -> None:
        """Check SSA form, topological order, names and types; raise ValueError if broken.

        Attributes must be normalized: ``ShapeError`` (a ValueError) names the node otherwise.
        """
        from tessera.ir.ops import ShapeError, get_op

        defined: set[int] = {id(v) for v in self.inputs}
        names: dict[str, Value] = {}

        def define(v: Value) -> None:
            other = names.setdefault(v.name, v)
            if other is not v:
                raise ValueError(f"two values are named {v.name!r}")

        for v in self.inputs:
            define(v)
            if v.producer is not None or v.is_constant:
                raise ValueError(f"graph input {v!r} has a producer or data")
        for node in self.nodes:
            for v in node.inputs:
                if v.is_constant:
                    define(v)
                elif id(v) not in defined:
                    raise ValueError(f"{node!r} reads {v!r} before it is defined")
            try:
                expected = get_op(node.op).infer([v.type for v in node.inputs], node.attrs)
            except ShapeError as e:
                raise ShapeError(f"{node!r}: {e}") from None
            if [v.type for v in node.outputs] != list(expected):
                got = [str(v.type) for v in node.outputs]
                raise ValueError(f"{node!r}: output types {got} != inferred {expected}")
            for v in node.outputs:
                if id(v) in defined or v.producer is not node:
                    raise ValueError(f"{v!r} is defined twice or has the wrong producer")
                define(v)
                defined.add(id(v))
        for v in self.outputs:
            if not (v.is_constant or id(v) in defined):
                raise ValueError(f"graph output {v!r} is never defined")

    def copy(self) -> Graph:
        """A deep copy of the structure; constant arrays are shared, not copied."""
        memo: dict[int, Any] = {}
        for v in self.values():
            if v.data is not None:
                memo[id(v.data)] = v.data
        return copy.deepcopy(self, memo)

    def __str__(self) -> str:
        lines = [f"graph {self.name}("]
        lines += [f"  {v!r}," for v in self.inputs]
        lines.append(") {")
        for node in self.nodes:
            lines.append(f"  {node!r}")
        lines.append(f"  return {', '.join(f'%{v.name}' for v in self.outputs)}")
        lines.append("}")
        return "\n".join(lines)
