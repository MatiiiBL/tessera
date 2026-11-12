"""Registers every operator (docs/design.md, section 3).

``CANONICAL_OPS`` are the operators a graph holds after ``passes.optimize``;
``IMPORT_OPS`` are ONNX operators the importer may leave behind and the
optimizer always removes.
"""

from __future__ import annotations

from tessera.ir.opdefs import anchor, broadcast, elementwise, injective, onnx_only, reduction

CANONICAL_OPS: frozenset[str] = frozenset(
    elementwise.NAMES + broadcast.NAMES + injective.NAMES + reduction.NAMES + anchor.NAMES
)
IMPORT_OPS: frozenset[str] = frozenset(onnx_only.NAMES)

__all__ = ["CANONICAL_OPS", "IMPORT_OPS"]
