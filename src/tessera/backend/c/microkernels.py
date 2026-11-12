"""Hand-written C functions that generated code may call.

A loop nest replaced by ``tensorize`` becomes ``Evaluate(Call("c.<name>",
args))``. Code generation looks the name up here to find the C function, the
header that defines it (shipped next to this module) and the least ISA that
can run it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources

from tessera.target import CpuIsa, host_isa

ISA_LEVEL: dict[CpuIsa, int] = {"generic": 0, "neon": 1, "sme": 2}


@dataclass(frozen=True)
class Microkernel:
    name: str  # the Call name, "c.<something>"
    symbol: str  # the C function
    header: str  # file under tessera/backend/c/include
    isa: CpuIsa  # least ISA that has the instructions it uses

    def available_on(self, isa: CpuIsa) -> bool:
        return ISA_LEVEL[isa] >= ISA_LEVEL[self.isa]


_REGISTRY: dict[str, Microkernel] = {}


def register_microkernel(kernel: Microkernel) -> Microkernel:
    """Make ``kernel`` callable from generated code (re-registering the same one is
    allowed; a different one under the same name is an error)."""
    existing = _REGISTRY.get(kernel.name)
    if existing is not None and existing != kernel:
        raise ValueError(f"microkernel {kernel.name!r} is already registered")
    _REGISTRY[kernel.name] = kernel
    return kernel


def get_microkernel(name: str) -> Microkernel | None:
    return _REGISTRY.get(name)


def host_supports(isa: CpuIsa) -> bool:
    """Whether this machine can run code built for ``isa``."""
    return ISA_LEVEL[host_isa()] >= ISA_LEVEL[isa]


# Implicit-GEMM convolution of one NHWC image with HWIO weights, on output
# pixels [p0, p0 + m) x channels [0, n): arguments ``(m, n, p0, x, w, ldw, y,
# ldy, h, w, c, kh, kw, sh, sw, ph, pw, dh, dw, ow, accumulate)``.
CONV2D_KERNELS = {
    isa: register_microkernel(
        Microkernel(f"c.{isa}_conv2d", f"tsr_{isa}_conv2d", f"tessera_{isa}.h", isa)
    )
    for isa in ("neon", "sme")
}

# GEMM on one tile, ``C[0:m, 0:n] (+)= A[0:m, 0:k] B[0:k, 0:n]``, with arguments
# ``(m, n, k, A, lda, B, ldb, C, ldc, accumulate)``: row-major operands with
# leading dimensions in elements; ``accumulate`` 0 overwrites C. The ``_bt``
# variants read B transposed, ``B[p, j]`` at ``B[j * ldb + p]``.
SGEMM_KERNELS = {
    isa: {
        transposed: register_microkernel(
            Microkernel(
                f"c.{isa}_sgemm{'_bt' if transposed else ''}",
                f"tsr_{isa}_sgemm{'_bt' if transposed else ''}",
                f"tessera_{isa}.h",
                isa,
            )
        )
        for transposed in (False, True)
    }
    for isa in ("neon", "sme")
}


def header_text(name: str) -> str:
    """The contents of a C header shipped with the backend."""
    return resources.files("tessera.backend.c").joinpath("include", name).read_text()


def header_files(names: Sequence[str]) -> dict[str, str]:
    """The headers ``names`` and every backend header they include, by name."""
    files: dict[str, str] = {}
    pending = list(names)
    while pending:
        name = pending.pop(0)
        if name in files:
            continue
        files[name] = header_text(name)
        pending += re.findall(r'^#include "([^"]+)"', files[name], re.MULTILINE)
    return files
