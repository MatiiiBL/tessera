"""The kernel shown in docs/metal.md is what the code generator prints today."""

from __future__ import annotations

import re
from pathlib import Path

from metal_workloads import softmax

from tessera.backend.metal.bench import schedule_default
from tessera.backend.metal.codegen import generate

DOC = Path(__file__).resolve().parents[2] / "docs" / "metal.md"


def test_documented_softmax_kernel_is_current() -> None:
    blocks = re.findall(r"```metal\n(.*?)```", DOC.read_text(), flags=re.DOTALL)
    func, _ = schedule_default(softmax(12, 128).lowered)
    assert blocks[0] == generate(func).source
