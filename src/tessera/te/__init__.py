"""Tensor expressions: declare computations, get PrimFuncs."""

from tessera.te.tensor import (
    ComputeOp,
    PlaceholderOp,
    Reduce,
    Tensor,
    compute,
    create_prim_func,
    max,
    min,
    placeholder,
    reduce_axis,
    sum,
)

__all__ = [
    "ComputeOp",
    "PlaceholderOp",
    "Reduce",
    "Tensor",
    "compute",
    "create_prim_func",
    "max",
    "min",
    "placeholder",
    "reduce_axis",
    "sum",
]
