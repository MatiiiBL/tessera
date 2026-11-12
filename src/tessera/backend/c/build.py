"""Compiling generated C with clang, cached by a hash of the sources and flags.

A build writes its sources into a fresh directory under the cache, compiles
there and renames the directory into place, so concurrent builds of the same
code (tuning builds candidates in parallel processes) never see a partial
file. The cache lives in ``$TESSERA_CACHE/c`` (default ``~/.cache/tessera/c``);
each entry keeps the sources next to the binary for inspection.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from tessera.target import Target

type OutputKind = Literal["shared", "executable"]

# -O3 with FMA contraction and reassociation (so reductions vectorize), but
# IEEE infinities and NaNs kept: max-reductions start from -INFINITY.
COMMON_FLAGS = (
    "-std=c11",
    "-O3",
    "-ffp-contract=fast",
    "-fno-math-errno",
    "-fno-trapping-math",
    "-fno-signed-zeros",
    "-fassociative-math",
    "-fvisibility=hidden",
    "-Wall",
    "-Wno-unused-function",
    "-Wno-unused-variable",
    "-Werror=implicit-function-declaration",
    "-Werror=incompatible-pointer-types",
)

# A build that takes longer than this is reported as failed.
BUILD_TIMEOUT = 600.0


class BuildError(Exception):
    """clang rejected the generated code; the message has its diagnostics."""


def cache_root() -> Path:
    """The root of Tessera's caches (``$TESSERA_CACHE`` or ``~/.cache/tessera``)."""
    return Path(os.environ.get("TESSERA_CACHE", "~/.cache/tessera")).expanduser()


def cache_dir() -> Path:
    return cache_root() / "c"


def isa_flags(target: Target) -> list[str]:
    """Machine flags for the target's ISA.

    ``generic`` adds none, so the code runs on any CPU clang targets by
    default. ``neon`` tunes for the host Arm core. ``sme`` targets the M4,
    whose SME and SME2 the microkernels use in streaming mode; the M4 has no
    SVE outside streaming mode and ``apple-m4`` does not enable it, so the
    ordinary code clang vectorizes stays NEON.
    """
    if target.isa == "generic":
        return []
    if target.isa == "neon":
        return ["-mcpu=native"]
    return ["-mcpu=apple-m4"]


def runtime_defines(target: Target) -> list[str]:
    """The ``#define`` lines a source puts before ``tessera_runtime.h``.

    They size the thread pool for the target. On SME targets they also make
    idle workers sleep at once instead of polling for the next loop: macOS
    moves threads that keep the shared SME unit busy onto the efficiency
    cluster, and a worker that never blocks stays there, three times slower,
    for the rest of the run (docs/cpu.md, "Threads on SME targets").
    """
    lines = [f"#define TSR_NUM_THREADS {target.threads}"]
    if target.isa == "sme":
        lines.append("#define TSR_SPIN_SECONDS 0")
    return lines


def compile_flags(target: Target, kind: OutputKind = "shared") -> list[str]:
    """Every flag of a build for ``target``, in order."""
    flags = [*COMMON_FLAGS, *isa_flags(target), *target.cflags]
    if kind == "shared":
        flags += ["-fPIC", "-shared"]
    return flags


def link_flags() -> list[str]:
    return [] if platform.system() == "Darwin" else ["-lm", "-lpthread"]


@functools.cache
def compiler_version(cc: str) -> str:
    """The compiler's first ``--version`` line (part of the cache key)."""
    try:
        out = subprocess.run(
            [cc, "--version"], capture_output=True, text=True, check=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise BuildError(f"cannot run the C compiler {cc!r}: {e}") from e
    return out.stdout.splitlines()[0] if out.stdout else cc


def build_key(sources: Mapping[str, str], flags: list[str], cc: str) -> str:
    """The cache key: a hash of the compiler, its flags and every source file."""
    blob = json.dumps(
        {"cc": compiler_version(cc), "flags": flags, "sources": sorted(sources.items())},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def output_name(kind: OutputKind) -> str:
    if kind == "executable":
        return "harness"
    return "libtessera.dylib" if platform.system() == "Darwin" else "libtessera.so"


def build(
    sources: Mapping[str, str],
    main: str,
    target: Target,
    *,
    kind: OutputKind = "shared",
) -> Path:
    """Compile ``sources[main]`` (other entries are headers next to it) for
    ``target`` and return the binary's path, reusing a cached build."""
    flags = compile_flags(target, kind)
    key = build_key(sources, flags, target.cc)
    root = cache_dir()
    entry = root / key
    binary = entry / output_name(kind)
    if binary.exists():
        return binary
    root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{key}-", dir=root))
    try:
        for name, text in sources.items():
            (work / name).write_text(text)
        cmd = [target.cc, *flags, "-o", output_name(kind), main, *link_flags()]
        try:
            result = subprocess.run(
                cmd, cwd=work, capture_output=True, text=True, timeout=BUILD_TIMEOUT
            )
        except subprocess.TimeoutExpired as e:
            raise BuildError(f"clang did not finish in {BUILD_TIMEOUT:.0f} s") from e
        except OSError as e:
            raise BuildError(f"cannot run the C compiler {target.cc!r}: {e}") from e
        if result.returncode != 0:
            raise BuildError(
                f"{' '.join(cmd)} failed (exit {result.returncode}):\n{result.stderr.strip()}"
            )
        (work / "build.json").write_text(json.dumps({"command": cmd}, indent=2) + "\n")
        try:
            work.rename(entry)
        except OSError:
            if not binary.exists():  # not a concurrent build of the same key
                raise
    finally:
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)
    return binary
