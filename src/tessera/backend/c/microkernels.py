"""Hand-written C functions that generated code may call.

A loop nest replaced by ``tensorize`` becomes ``Evaluate(Call("c.<name>",
args))``. Code generation looks the name up here to find the C function, the
header that defines it (shipped next to this module) and the least ISA that
can run it.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

from tessera.target import CpuIsa

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


def header_text(name: str) -> str:
    """The contents of a C header shipped with the backend."""
    return resources.files("tessera.backend.c").joinpath("include", name).read_text()
