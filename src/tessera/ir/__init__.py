"""The graph IR: operators on tensors with static shapes."""

from tessera.ir.graph import Attr, Graph, Node, TensorType, Value
from tessera.ir.interpret import interpret, random_inputs
from tessera.ir.ops import (
    OpDef,
    OpKind,
    ShapeError,
    get_op,
    normalize_attrs,
    register_op,
    registered_ops,
)

__all__ = [
    "Attr",
    "Graph",
    "Node",
    "OpDef",
    "OpKind",
    "ShapeError",
    "TensorType",
    "Value",
    "get_op",
    "interpret",
    "normalize_attrs",
    "random_inputs",
    "register_op",
    "registered_ops",
]
