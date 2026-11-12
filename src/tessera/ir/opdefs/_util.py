"""Attribute specifications, normalization and shape checks shared by the operators.

Each operator declares its attributes with ``Spec``. ``normalize`` accepts the
lenient spelling (defaults omitted, lists, ints for floats, a scalar for a
per-dimension tuple) and returns the canonical one; ``check`` accepts only the
canonical one. Shape inference calls ``check`` first, so a node built without
normalization is rejected with a message that says what is wrong.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from tessera.dtype import DType
from tessera.ir.graph import Attr, NDArray, TensorType
from tessera.ir.ops import InferFn, NormalizeFn, OpDef, OpKind, ReferenceFn, ShapeError, register_op

type AttrKind = Literal["int", "float", "str", "ints", "floats"]
type Specs = Mapping[str, Spec]


@dataclass(frozen=True)
class Spec:
    """One attribute: its kind, its default (None: required) and allowed values."""

    kind: AttrKind
    default: Attr | None = None
    choices: tuple[Attr, ...] = ()
    length: int | None = None  # for ints/floats: a scalar is repeated to this length


def _coerce(key: str, spec: Spec, value: object) -> Attr:
    """The canonical Python type of an attribute value given leniently."""
    try:
        match spec.kind:
            case "int":
                if isinstance(value, float) and not value.is_integer():
                    raise TypeError
                return int(value)  # type: ignore[call-overload,no-any-return]
            case "float":
                return float(value)  # type: ignore[arg-type]
            case "str":
                if not isinstance(value, str):
                    raise TypeError
                return str(value)
            case "ints":
                if isinstance(value, int | np.integer):
                    if spec.length is None:
                        raise TypeError
                    return (int(value),) * spec.length
                items = list(value)  # type: ignore[call-overload]
                if any(isinstance(v, float) and not v.is_integer() for v in items):
                    raise TypeError
                return tuple(int(v) for v in items)
            case "floats":
                if isinstance(value, int | float | np.number):
                    if spec.length is None:
                        raise TypeError
                    return (float(value),) * spec.length
                return tuple(float(v) for v in value)  # type: ignore[attr-defined]
    except (TypeError, ValueError):
        pass
    raise ShapeError(f"attribute {key!r} must be {_describe(spec)}, got {value!r}")


def _describe(spec: Spec) -> str:
    return {
        "int": "an int",
        "float": "a number",
        "str": "a string",
        "ints": "a tuple of ints",
        "floats": "a tuple of numbers",
    }[spec.kind]


def _canonical_type(spec: Spec, value: Attr) -> bool:
    match spec.kind:
        case "int":
            return isinstance(value, int) and not isinstance(value, bool)
        case "float":
            return isinstance(value, float)
        case "str":
            return isinstance(value, str)
        case "ints":
            return isinstance(value, tuple) and all(
                isinstance(v, int) and not isinstance(v, bool) for v in value
            )
        case "floats":
            return isinstance(value, tuple) and all(isinstance(v, float) for v in value)


def normalize_specs(specs: Specs, attrs: Mapping[str, Attr]) -> dict[str, Attr]:
    """Fill defaults and coerce types; reject unknown or missing attributes."""
    unknown = sorted(set(attrs) - set(specs))
    if unknown:
        raise ShapeError(f"unknown attribute(s) {', '.join(map(repr, unknown))}")
    out: dict[str, Attr] = {}
    for key, spec in specs.items():
        if key in attrs:
            value = _coerce(key, spec, attrs[key])
        elif spec.default is not None:
            value = spec.default
        else:
            raise ShapeError(f"missing required attribute {key!r}")
        if spec.choices and value not in spec.choices:
            raise ShapeError(f"attribute {key!r} must be one of {spec.choices}, got {value!r}")
        out[key] = value
    return out


def check_specs(specs: Specs, attrs: Mapping[str, Attr]) -> None:
    """Require exactly the normalized attributes (every one present, canonical types)."""
    unknown = sorted(set(attrs) - set(specs))
    if unknown:
        raise ShapeError(f"unknown attribute(s) {', '.join(map(repr, unknown))}")
    for key, spec in specs.items():
        if key not in attrs:
            raise ShapeError(
                f"attribute {key!r} is missing (Graph.add_node normalizes attributes; "
                "nodes built directly must spell every attribute)"
            )
        value = attrs[key]
        if not _canonical_type(spec, value):
            raise ShapeError(f"attribute {key!r} must be {_describe(spec)}, got {value!r}")
        if spec.choices and value not in spec.choices:
            raise ShapeError(f"attribute {key!r} must be one of {spec.choices}, got {value!r}")
        if spec.length is not None and isinstance(value, tuple) and len(value) != spec.length:
            raise ShapeError(f"attribute {key!r} must have {spec.length} entries, got {value!r}")


def ints(attrs: Mapping[str, Attr], key: str) -> tuple[int, ...]:
    """A tuple-of-ints attribute that ``check_specs`` has validated."""
    value = attrs[key]
    assert isinstance(value, tuple)
    return tuple(int(v) for v in value)


def integer(attrs: Mapping[str, Attr], key: str) -> int:
    value = attrs[key]
    assert isinstance(value, int)
    return value


def number(attrs: Mapping[str, Attr], key: str) -> float:
    value = attrs[key]
    assert isinstance(value, int | float)
    return float(value)


def string(attrs: Mapping[str, Attr], key: str) -> str:
    value = attrs[key]
    assert isinstance(value, str)
    return value


# -- shape checks -------------------------------------------------------------


def arity(types: Sequence[TensorType], lo: int, hi: int | None = None) -> None:
    """Require between ``lo`` and ``hi`` inputs (exactly ``lo`` when ``hi`` is None)."""
    hi = lo if hi is None else hi
    if not lo <= len(types) <= hi:
        want = str(lo) if lo == hi else f"{lo} to {hi}"
        raise ShapeError(f"expects {want} inputs, got {len(types)}")


def require_float(t: TensorType, what: str = "input") -> None:
    if not t.dtype.is_float:
        raise ShapeError(f"{what} must be a float tensor, got {t}")


def require_numeric(t: TensorType, what: str = "input") -> None:
    if t.dtype is DType.bool_:
        raise ShapeError(f"{what} must be numeric, got {t}")


def require_rank(t: TensorType, rank: int, what: str = "input") -> None:
    if t.rank != rank:
        raise ShapeError(f"{what} must have rank {rank}, got {t}")


def same_dtype(types: Sequence[TensorType], what: str = "inputs") -> DType:
    dtypes = {t.dtype for t in types}
    if len(dtypes) != 1:
        raise ShapeError(f"{what} must have one element type, got {', '.join(map(str, types))}")
    return types[0].dtype


def broadcast_shapes(shapes: Sequence[tuple[int, ...]]) -> tuple[int, ...]:
    """NumPy broadcasting of ``shapes``; ``ShapeError`` if they do not broadcast."""
    rank = max((len(s) for s in shapes), default=0)
    out: list[int] = []
    for i in range(rank):
        dims = {s[len(s) - rank + i] for s in shapes if len(s) - rank + i >= 0}
        dims.discard(1)
        if len(dims) > 1:
            listed = ", ".join(str(list(s)) for s in shapes)
            raise ShapeError(f"shapes {listed} do not broadcast")
        out.append(dims.pop() if dims else 1)
    return tuple(out)


def normalize_axis(axis: int, rank: int, what: str = "axis") -> int:
    """``axis`` in ``[-rank, rank)`` made non-negative."""
    if not -rank <= axis < rank:
        raise ShapeError(f"{what} {axis} is out of range for rank {rank}")
    return axis + rank if axis < 0 else axis


def check_axis(axis: int, rank: int, what: str = "axis") -> None:
    """A normalized axis: in ``[0, rank)``."""
    if not 0 <= axis < rank:
        raise ShapeError(f"{what} {axis} must be normalized to [0, {rank})")


def check_axes(axes: Sequence[int], rank: int, what: str = "axes") -> None:
    """Normalized axes: sorted, unique, in ``[0, rank)``."""
    if list(axes) != sorted(set(axes)) or any(not 0 <= a < rank for a in axes):
        raise ShapeError(f"{what} {tuple(axes)} must be sorted, unique and in [0, {rank})")


def check_perm(perm: Sequence[int], rank: int) -> None:
    if sorted(perm) != list(range(rank)):
        raise ShapeError(f"perm {tuple(perm)} is not a permutation of {rank} axes")


def axis_normalizer(specs: Specs, key: str = "axis") -> NormalizeFn:
    """A normalizer that makes the int attribute ``key`` an axis of the first input."""

    def normalize(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> dict[str, Attr]:
        out = normalize_specs(specs, attrs)
        if types:
            out[key] = normalize_axis(integer(out, key), types[0].rank)
        return out

    return normalize


# -- registration -------------------------------------------------------------


def define(
    name: str,
    kind: OpKind,
    infer: Callable[[Sequence[TensorType], Mapping[str, Attr]], list[TensorType]],
    reference: ReferenceFn,
    *,
    specs: Specs | None = None,
    normalize: NormalizeFn | None = None,
    doc: str = "",
) -> OpDef:
    """Register an operator whose normalized attributes are described by ``specs``.

    The registered ``infer`` checks the attributes against ``specs`` before
    calling ``infer``. Attributes are normalized by ``normalize`` when given
    (for rank-dependent rules such as negative axes; it sees the attributes as
    written), else by applying ``specs``.
    """
    canonical = dict(specs or {})

    def checked_infer(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> list[TensorType]:
        check_specs(canonical, attrs)
        return infer(types, attrs)

    def spec_normalize(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> dict[str, Attr]:
        return normalize_specs(canonical, attrs)

    infer_fn: InferFn = checked_infer
    return register_op(OpDef(name, kind, infer_fn, reference, doc, normalize or spec_normalize))


def keep_dtype(x: NDArray, like: NDArray) -> NDArray:
    """``x`` with ``like``'s element type (NumPy promotes some scalar mixes)."""
    return x if x.dtype == like.dtype else x.astype(like.dtype)
