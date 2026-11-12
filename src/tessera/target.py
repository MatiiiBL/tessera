"""Compilation targets: which backend, and what the hardware offers."""

from __future__ import annotations

import functools
import os
import platform
import subprocess
from dataclasses import dataclass, replace
from typing import Literal

type BackendKind = Literal["c", "metal"]
type CpuIsa = Literal["generic", "neon", "sme"]


@dataclass(frozen=True)
class Target:
    """What to compile for.

    ``kind`` picks the backend. For the C backend, ``isa`` says which
    microkernels may be used: ``generic`` is portable C (any machine, CI on
    x86), ``neon`` adds Arm NEON intrinsics, ``sme`` adds the Scalable Matrix
    Extension (Apple M4). ``num_threads`` 0 means one per performance core.
    """

    kind: BackendKind = "c"
    isa: CpuIsa = "generic"
    num_threads: int = 0
    cc: str = "clang"
    cflags: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        """Stable identity for caches and tuning logs (hardware included)."""
        parts = [self.kind, self.isa, f"t{self.threads}", _cpu_name()]
        return "-".join(p.replace(" ", "_") for p in parts)

    @property
    def threads(self) -> int:
        return self.num_threads or performance_cores()

    def with_(self, **changes: object) -> Target:
        return replace(self, **changes)  # type: ignore[arg-type]

    @classmethod
    def host(cls, kind: BackendKind = "c") -> Target:
        """The best target for this machine."""
        return cls(kind=kind, isa=host_isa())

    @classmethod
    def parse(cls, text: str) -> Target:
        """``c``, ``c:neon``, ``c:sme``, ``c:generic``, ``metal`` or ``host``."""
        if text == "host":
            return cls.host()
        kind, _, isa = text.partition(":")
        if kind not in ("c", "metal"):
            raise ValueError(f"unknown target {text!r} (c, c:<isa>, metal, host)")
        if kind == "metal":
            return cls(kind="metal", isa=host_isa())
        if not isa:
            return cls.host()
        if isa not in ("generic", "neon", "sme"):
            raise ValueError(f"unknown ISA {isa!r} (generic, neon, sme)")
        return cls(kind="c", isa=isa)  # type: ignore[arg-type]


def host_isa() -> CpuIsa:
    if platform.machine() not in ("arm64", "aarch64"):
        return "generic"
    if platform.system() == "Darwin" and _sysctl("hw.optional.arm.FEAT_SME") == "1":
        return "sme"
    return "neon"


def performance_cores() -> int:
    if platform.system() == "Darwin":
        value = _sysctl("hw.perflevel0.physicalcpu")
        if value and value.isdigit():
            return int(value)
    return os.cpu_count() or 1


def _cpu_name() -> str:
    if platform.system() == "Darwin":
        return _sysctl("machdep.cpu.brand_string") or platform.machine()
    return platform.machine()


@functools.cache
def _sysctl(name: str) -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()
