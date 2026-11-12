"""Shape inference rejects bad inputs and unnormalized attributes with clear messages."""

from __future__ import annotations

import numpy as np
import pytest

from tessera.dtype import DType
from tessera.ir import Graph, Node, ShapeError, TensorType, Value, get_op, normalize_attrs
from tessera.ir.opdefs import CANONICAL_OPS, IMPORT_OPS

F32 = DType.float32


def t(*shape: int, dtype: DType = F32) -> TensorType:
    return TensorType(shape, dtype)


def build(op: str, types: list[TensorType], **attrs: object) -> TensorType:
    g = Graph()
    values = [g.add_input(f"x{i}", ty) for i, ty in enumerate(types)]
    return g.op(op, *values, **attrs).type  # type: ignore[arg-type]


CASES = [
    ("MatMul", [t(2, 3), t(4, 5)], {}, "inner dimensions differ"),
    ("MatMul", [t(3), t(3, 4)], {}, "rank >= 2"),
    ("MatMul", [t(2, 2, 3), t(3, 3, 4)], {}, "do not broadcast"),
    ("Add", [t(2, 3), t(4)], {}, "do not broadcast"),
    ("Add", [t(2), t(2, dtype=DType.int64)], {}, "one element type"),
    ("Add", [t(2, dtype=DType.bool_), t(2, dtype=DType.bool_)], {}, "numeric"),
    ("Exp", [t(2, dtype=DType.int64)], {}, "float tensor"),
    ("Relu", [t(2), t(2)], {}, "expects 1 inputs"),
    ("Where", [t(2), t(2), t(2)], {}, "condition must be bool"),
    ("Reshape", [t(2, 3)], {"shape": (4, 2)}, "cannot reshape"),
    ("Reshape", [t(2, 3)], {"shape": (-1, -1)}, "more than one -1"),
    ("Reshape", [t(2, 3)], {"shape": (5, -1)}, "cannot reshape"),
    ("Reshape", [t(2, 3)], {}, "missing required attribute 'shape'"),
    ("Transpose", [t(2, 3)], {"perm": (0, 0)}, "not a permutation"),
    ("Transpose", [t(2, 3)], {"perm": (0, 2)}, "out of range"),
    ("Concat", [t(2, 3), t(2, 4)], {"axis": 0}, "differ outside axis 0"),
    ("Concat", [t(2, 3), t(2, 4, 1)], {"axis": 1}, "differ outside axis"),
    ("Concat", [t(2, 3)], {"axis": 2}, "out of range"),
    ("Gather", [t(4, 3), t(2)], {}, "indices must be int64 or int32"),
    ("Slice", [t(4, 3)], {"starts": (0,), "ends": (2,), "steps": (-1,)}, "positive steps"),
    ("Slice", [t(4, 3)], {"starts": (0, 0), "ends": (2,)}, "differ in length"),
    ("Expand", [t(3, 2)], {"shape": (4, 3)}, "do not broadcast"),
    ("Pad", [t(3, 2)], {"pads": (1, 1)}, "2 x rank"),
    ("Pad", [t(3, 2)], {"pads": (1, 1, -1, 0)}, "non-negative"),
    ("Softmax", [t(3, 2)], {"axis": 2}, "out of range"),
    ("ReduceSum", [t(3, 2)], {"axes": (0, -2)}, "repeat an axis"),
    ("LayerNorm", [t(2, 8), t(4), t(8)], {}, "scale must have shape (8,)"),
    ("Conv2d", [t(1, 3, 8, 8), t(4, 2, 3, 3)], {}, "needs 3"),
    ("Conv2d", [t(1, 4, 8, 8), t(4, 2, 3, 3)], {"groups": 3}, "must divide"),
    ("Conv2d", [t(1, 3, 2, 2), t(4, 3, 3, 3)], {}, "does not fit"),
    ("Conv2d", [t(1, 3, 8, 8), t(4, 3, 3, 3), t(5)], {}, "bias must have shape (4,)"),
    ("Conv2d", [t(3, 8, 8), t(4, 3, 3, 3)], {}, "rank 4"),
    ("Conv2d", [t(1, 3, 8, 8), t(4, 3, 3, 3)], {"layout": "NCWH"}, "must be one of"),
    ("MaxPool2d", [t(1, 3, 8, 8)], {"kernel": (2, 2), "pads": (2, 2)}, "smaller than the kernel"),
    ("MaxPool2d", [t(1, 3, 8, 8)], {"kernel": (2, 2), "strides": (0, 1)}, ">= 1"),
    ("AvgPool2d", [t(1, 3, 8, 8)], {}, "missing required attribute 'kernel'"),
    ("Gemm", [t(2, 3), t(2, 3)], {}, "inner dimensions differ"),
    ("Squeeze", [t(1, 3)], {"axes": (1,)}, "dimensions must be 1"),
    ("Gelu", [t(3)], {"approximate": "fast"}, "must be one of"),
    ("Cast", [t(3)], {"to": "float64"}, "must be one of"),
    ("Relu", [t(3)], {"alpha": 1.0}, "unknown attribute"),
    ("BatchNorm", [t(2, 3), t(3), t(3), t(2), t(3)], {}, "mean must have shape (3,)"),
]


@pytest.mark.parametrize(("op", "types", "attrs", "message"), CASES)
def test_shape_inference_rejects(
    op: str, types: list[TensorType], attrs: dict[str, object], message: str
) -> None:
    with pytest.raises(ShapeError, match=message.replace("(", r"\(").replace(")", r"\)")):
        build(op, types, **attrs)


def test_error_names_the_node() -> None:
    g = Graph()
    a = g.add_input("a", t(2, 3))
    with pytest.raises(ShapeError, match="MatMul node 'proj'"):
        g.op("MatMul", a, a, name="proj")


def test_infer_requires_normalized_attributes() -> None:
    softmax = get_op("Softmax")
    assert softmax.infer([t(2, 3)], {"axis": 1}) == [t(2, 3)]
    with pytest.raises(ShapeError, match="normalized"):
        softmax.infer([t(2, 3)], {"axis": -1})
    with pytest.raises(ShapeError, match="missing"):
        get_op("Gelu").infer([t(2)], {})
    with pytest.raises(ShapeError, match="must be a number"):
        get_op("Clip").infer([t(2)], {"min": 0, "max": 1.0})


def test_verify_rejects_unnormalized_node() -> None:
    g = Graph()
    x = g.add_input("x", t(2, 3))
    out = Value("y", t(2, 3))
    g.nodes.append(Node("Softmax", [x], [out], {"axis": -1}, "y"))
    g.set_outputs([out])
    with pytest.raises(ShapeError, match="Softmax"):
        g.verify()


NORMALIZED = [
    ("Softmax", [t(2, 3)], {}, {"axis": 1}),
    ("ReduceMean", [t(2, 3, 4)], {"axes": (-1, 0)}, {"axes": (0, 2), "keepdims": 1}),
    ("ReduceMax", [t(2, 3)], {"keepdims": 0}, {"axes": (0, 1), "keepdims": 0}),
    ("Reshape", [t(2, 3, 4)], {"shape": [0, -1]}, {"shape": (2, 12)}),
    ("Transpose", [t(2, 3, 4)], {}, {"perm": (2, 1, 0)}),
    ("Expand", [t(3, 1)], {"shape": (2, 1, 4)}, {"shape": (2, 3, 4)}),
    (
        "Slice",
        [t(4, 6)],
        {"starts": [-2], "ends": [10**9], "axes": [1]},
        {"starts": (0, 4), "ends": (4, 6), "steps": (1, 1)},
    ),
    ("Concat", [t(2, 3), t(2, 3)], {"axis": -1}, {"axis": 1}),
    ("Gather", [t(5, 2), t(3, dtype=DType.int64)], {"axis": -2}, {"axis": 0}),
    ("Clip", [t(3)], {"min": 0}, {"min": 0.0, "max": float("inf")}),
    ("Squeeze", [t(1, 3, 1)], {}, {"axes": (0, 2)}),
    ("Unsqueeze", [t(3)], {"axes": [-1, 0]}, {"axes": (0, 2)}),
    ("Flatten", [t(2, 3, 4)], {"axis": -1}, {"axis": 2}),
    (
        "MaxPool2d",
        [t(1, 1, 4, 4)],
        {"kernel": 2, "pads": [1, 0]},
        {"kernel": (2, 2), "strides": (1, 1), "pads": (1, 0, 1, 0), "layout": "NCHW"},
    ),
]


@pytest.mark.parametrize(("op", "types", "attrs", "expected"), NORMALIZED)
def test_normalize(
    op: str, types: list[TensorType], attrs: dict[str, object], expected: dict[str, object]
) -> None:
    got = normalize_attrs(op, types, attrs)  # type: ignore[arg-type]
    assert got == expected
    get_op(op).infer(types, got)


def test_every_operator_has_a_reference_and_docs() -> None:
    assert CANONICAL_OPS.isdisjoint(IMPORT_OPS)
    for name in CANONICAL_OPS | IMPORT_OPS:
        opdef = get_op(name)
        assert opdef.doc, name
        assert callable(opdef.reference)


def test_design_operator_set_is_registered() -> None:
    design = {
        *("Relu", "Sigmoid", "Tanh", "Exp", "Log", "Sqrt", "Rsqrt", "Erf", "Neg", "Abs"),
        *("Silu", "HardSwish", "HardSigmoid", "Gelu", "Clip", "Cast"),
        *("Add", "Sub", "Mul", "Div", "Pow", "Maximum", "Minimum", "Equal", "Less", "Greater"),
        *("Where", "BatchNorm", "Reshape", "Transpose", "Slice", "Concat", "Gather", "Expand"),
        *("Pad", "ReduceSum", "ReduceMean", "ReduceMax", "Softmax", "LayerNorm", "MaxPool2d"),
        *("AvgPool2d", "Conv2d", "MatMul"),
    }
    assert design == CANONICAL_OPS


def test_float64_inputs_keep_float64_in_references() -> None:
    x = np.linspace(-3, 3, 7)
    for name in ("Gelu", "Sigmoid", "Erf", "Softmax", "Silu"):
        attrs = normalize_attrs(name, [t(7)], {})
        (y,) = get_op(name).reference([x], attrs)
        assert y.dtype == np.float64, name
