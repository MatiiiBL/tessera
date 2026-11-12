"""The operator registry.

Every operator the graph IR can hold is registered once with its shape
inference, a NumPy reference implementation and its fusion kind. The
reference implementations define the semantics: the graph interpreter runs
them, and every compiled kernel is tested against them.

The operator set and the attributes of each operator are listed in
docs/design.md (section 3); ``tessera.ir.opdefs`` registers them.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from tessera.ir.graph import Attr, NDArray, TensorType


class OpKind(enum.IntEnum):
    """How an operator's output elements depend on its inputs (decides fusion).

    The order matters: fusion merges a producer into a consumer only if the
    producer's kind is at most INJECTIVE, and a group holds at most one node
    of kind REDUCTION or ANCHOR.
    """

    ELEMWISE = 0  # out[i] = f(a[i], b[i], ...), all shapes equal
    BROADCAST = 1  # out[i] = f(a[bcast(i)], b[bcast(i)], ...)
    INJECTIVE = 2  # out[i] = a[g(i)]: Reshape, Transpose, Slice, Concat, Gather, Pad
    REDUCTION = 3  # ReduceSum, Softmax, LayerNorm, pooling
    ANCHOR = 4  # Conv2d, MatMul: the heavy op a template is built around
    OPAQUE = 5  # never fused


type InferFn = Callable[[Sequence[TensorType], Mapping[str, Attr]], list[TensorType]]
type ReferenceFn = Callable[[Sequence[NDArray], Mapping[str, Attr]], list[NDArray]]
type NormalizeFn = Callable[[Sequence[TensorType], Mapping[str, Attr]], dict[str, Attr]]


class ShapeError(ValueError):
    """Shape inference rejected an operator's inputs or attributes."""


@dataclass(frozen=True)
class OpDef:
    """One operator: its name, fusion kind, shape inference and reference.

    ``infer`` accepts only normalized attributes (every attribute present,
    axes non-negative, one entry per dimension) and raises ``ShapeError``
    otherwise. ``normalize``, when given, turns the lenient spelling a
    builder or importer uses (defaults omitted, negative axes, lists) into
    that form; ``Graph.add_node`` applies it.
    """

    name: str
    kind: OpKind
    infer: InferFn
    reference: ReferenceFn
    doc: str = ""
    normalize: NormalizeFn | None = None


_REGISTRY: dict[str, OpDef] = {}


def register_op(opdef: OpDef) -> OpDef:
    """Add an operator to the registry (each name once)."""
    if opdef.name in _REGISTRY:
        raise ValueError(f"operator {opdef.name} is registered twice")
    _REGISTRY[opdef.name] = opdef
    return opdef


def get_op(name: str) -> OpDef:
    """The definition of operator ``name``; ``KeyError`` if there is none."""
    _load_definitions()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown operator {name!r}") from None


def normalize_attrs(
    name: str, types: Sequence[TensorType], attrs: Mapping[str, Attr]
) -> dict[str, Attr]:
    """The normalized attributes of operator ``name`` applied to inputs of ``types``."""
    opdef = get_op(name)
    if opdef.normalize is None:
        return dict(attrs)
    return opdef.normalize(types, attrs)


def registered_ops() -> list[str]:
    """Names of every registered operator, sorted."""
    _load_definitions()
    return sorted(_REGISTRY)


def _load_definitions() -> None:
    # The definitions import this module, so they are loaded on first use.
    import tessera.ir.opdefs  # noqa: F401
