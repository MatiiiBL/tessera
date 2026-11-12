"""Writing benchmark results: ``results/<name>.json`` and ``results/<name>.md``.

Every result file records when and where it was measured: the date, the
machine, the commit, and the load average when the run started. A run
started on a busy machine (a one-minute load average above
``BUSY_LOAD``) is marked provisional, and its Markdown says so at the top.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tessera.target import performance_cores

BUSY_LOAD = 2.0


def repo_root() -> Path:
    """The checkout this package runs from (``src/tessera/bench`` is three levels down)."""
    return Path(__file__).resolve().parents[3]


def results_dir() -> Path:
    """``$TESSERA_RESULTS`` or ``results/`` in the checkout."""
    return Path(os.environ.get("TESSERA_RESULTS", repo_root() / "results"))


def environment(*, provisional: bool | None = None) -> dict[str, Any]:
    """Where and when a benchmark ran; ``provisional`` defaults to "the machine is busy"."""
    load = os.getloadavg()
    return {
        "date": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "machine": _sysctl("machdep.cpu.brand_string") or platform.machine(),
        "performance_cores": performance_cores(),
        "cores": os.cpu_count(),
        "memory_gb": _memory_gb(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "commit": _commit(),
        "load_average": [round(x, 2) for x in load],
        "provisional": load[0] > BUSY_LOAD if provisional is None else provisional,
    }


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """A GitHub Markdown table; numbers are written as given."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def header(title: str, env: dict[str, Any], command: str) -> str:
    """The top of a results page: title, provenance, provisional warning."""
    lines = [f"# {title}", ""]
    if env.get("provisional"):
        lines += [
            f"**Provisional:** measured while the machine was busy (load average "
            f"{env['load_average'][0]} at the start). Re-run on a quiet machine for final numbers.",
            "",
        ]
    lines += [
        f"Measured {env['date']} on {env['machine']} ({env['performance_cores']} performance "
        f"cores, {env['cores']} cores, {env['memory_gb']} GB), {env['os']}, commit "
        f"`{env['commit']}`. Reproduce with `{command}`.",
        "",
    ]
    return "\n".join(lines)


def write(name: str, data: dict[str, Any], markdown: str, directory: Path | None = None) -> Path:
    """Write ``<name>.json`` and ``<name>.md``; returns the Markdown path."""
    out = directory if directory is not None else results_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    md = out / f"{name}.md"
    md.write_text(markdown.rstrip() + "\n")
    return md


def _commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
            cwd=repo_root(),
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip()


def _sysctl(name: str) -> str | None:
    if platform.system() != "Darwin":
        return None
    try:
        out = subprocess.run(
            ["sysctl", "-n", name], capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _memory_gb() -> int | None:
    value = _sysctl("hw.memsize")
    if value and value.isdigit():
        return int(value) // 2**30
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2**30
    except (ValueError, OSError):
        return None
