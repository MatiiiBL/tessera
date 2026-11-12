"""Every group of the benchmark models lowers, verifies and takes simple CPU and GPU
schedules; small groups are interpreted against the operator references.

Run with ``uv run pytest -m models tests/compute``; the models come from
``scripts/export_models.py`` (``data/models``, or ``$TESSERA_MODELS``).
"""

from __future__ import annotations

import json
import os
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tessera import tir
from tessera.compute import lower_group
from tessera.frontend import onnx as onnx_frontend
from tessera.passes import fuse, optimize
from tessera.plan import FusedGroup
from tessera.tir.expr import IntImm
from tessera.tir.primitives.common import iter_paths
from tessera.tir.stmt import For, PrimFunc, Stmt

from . import schedules
from .harness import group_reference, values_of

pytestmark = pytest.mark.models

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("TESSERA_MODELS", ROOT / "data" / "models"))
NAMES = ["resnet50", "mobilenet_v2", "bert", "gpt2"]
# Groups whose loop nest has at most this many iterations are also interpreted.
INTERPRET_LIMIT = 2_000_000


@cache
def _model(name: str) -> tuple[list[FusedGroup], dict[str, Any]]:
    manifest = MODELS_DIR / "manifest.json"
    if not manifest.exists():
        pytest.skip(f"no exported models in {MODELS_DIR} (run scripts/export_models.py)")
    entries: dict[str, Any] = json.loads(manifest.read_text())
    if name not in entries:
        pytest.skip(f"{name} is not exported")
    entry = entries[name]
    data = np.load(MODELS_DIR / entry["npz"])
    inputs = {i["name"]: data[i["name"]] for i in entry["inputs"]}
    graph = optimize(onnx_frontend.load(MODELS_DIR / entry["onnx"]))
    groups = [g for g in fuse(graph) if not g.is_view]
    return groups, values_of(graph, inputs)


def _iterations(func: PrimFunc) -> int:
    """Iterations of all outermost loop nests (the work of an unscheduled group)."""
    total = 0
    for path in iter_paths(func.body):
        loop = path[-1]
        if isinstance(loop, For) and not any(isinstance(n, For) for n in path[:-1]):
            count = 1
            node: Stmt = loop
            while isinstance(node, For):
                assert isinstance(node.extent, IntImm)
                count *= node.extent.value
                node = node.body
            total += count
    return total


@pytest.mark.parametrize("name", NAMES)
def test_every_group_lowers_and_schedules(name: str) -> None:
    groups, values = _model(name)
    failures: list[str] = []
    interpreted = 0
    for group in groups:
        try:
            lowered = lower_group(group)
            tir.verify(lowered.func)
            funcs = []
            for make in (schedules.cpu, schedules.gpu):
                s = make(lowered)
                tir.verify(s.func)
                funcs.append(s.func)
        except Exception as e:  # collect every failing group, then report them together
            failures.append(f"{group.name} ({group.pattern}): {type(e).__name__}: {e}")
            continue
        if _iterations(lowered.func) > INTERPRET_LIMIT:
            continue
        expected = group_reference(group, values)
        args = [np.array(values[v.name], order="C") for v in group.inputs]
        for func in funcs:
            out = np.full(group.output.shape, 0, group.output.dtype.numpy)
            tir.interpret(func, [*args, out])
            scale = max(1.0, float(np.abs(expected).max(initial=0.0)))
            if not np.allclose(out, expected, rtol=2e-4, atol=2e-4 * scale):
                failures.append(f"{group.name} ({group.pattern}): wrong result")
        interpreted += 1
    assert not failures, "\n".join(failures)
    assert interpreted > 0
