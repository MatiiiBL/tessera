"""Registers every operator (docs/design.md, section 3).

Placeholder definitions of two operators, so the IR can be exercised before
the full operator set lands; ``tessera.ir.opdefs`` is owned by the graph
package.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from tessera.ir.graph import Attr, TensorType
from tessera.ir.ops import OpDef, OpKind, register_op


def _same(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> list[TensorType]:
    return [types[0]]


def _broadcast(types: Sequence[TensorType], attrs: Mapping[str, Attr]) -> list[TensorType]:
    shape = np.broadcast_shapes(*(t.shape for t in types))
    return [TensorType(tuple(int(d) for d in shape), types[0].dtype)]


register_op(
    OpDef(
        "Relu",
        OpKind.ELEMWISE,
        _same,
        lambda xs, attrs: [np.maximum(xs[0], 0).astype(xs[0].dtype)],
    )
)
register_op(
    OpDef(
        "Add",
        OpKind.BROADCAST,
        _broadcast,
        lambda xs, attrs: [np.add(xs[0], xs[1]).astype(xs[0].dtype)],
    )
)
