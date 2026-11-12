"""Element types shared by the graph IR, the loop IR and every backend."""

from __future__ import annotations

import enum
from typing import Any

import numpy as np
import numpy.typing as npt


class DType(enum.StrEnum):
    """The element types Tessera knows about.

    Kernels compute in ``float32``. The integer types carry indices (``Gather``),
    shapes and loop variables; ``bool_`` carries masks (``Where``).
    """

    float32 = "float32"
    float16 = "float16"
    int64 = "int64"
    int32 = "int32"
    int8 = "int8"
    uint8 = "uint8"
    bool_ = "bool"

    @property
    def itemsize(self) -> int:
        return _ITEMSIZE[self]

    @property
    def numpy(self) -> np.dtype[Any]:
        return np.dtype(self.value)

    @property
    def is_float(self) -> bool:
        return self in (DType.float32, DType.float16)

    @property
    def is_int(self) -> bool:
        return self in (DType.int64, DType.int32, DType.int8, DType.uint8)

    @classmethod
    def from_numpy(cls, dtype: npt.DTypeLike) -> DType:
        name = np.dtype(dtype).name
        try:
            return cls(name)
        except ValueError:
            raise ValueError(f"unsupported element type {name}") from None


_ITEMSIZE = {
    DType.float32: 4,
    DType.float16: 2,
    DType.int64: 8,
    DType.int32: 4,
    DType.int8: 1,
    DType.uint8: 1,
    DType.bool_: 1,
}

# Loop variables and index arithmetic. Backends may narrow it (Metal uses 32 bits)
# when every extent fits.
INDEX = DType.int64
