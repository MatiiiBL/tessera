"""CPU schedule templates, one per pattern of ``tessera.plan.PATTERNS``."""

from __future__ import annotations

import functools

from tessera.backend.base import Template
from tessera.backend.c.templates.common import CpuTarget
from tessera.backend.c.templates.conv import (
    schedule_conv2d,
    schedule_depthwise_conv2d,
    schedule_pool,
)
from tessera.backend.c.templates.elementwise import schedule_injective
from tessera.backend.c.templates.matmul import schedule_matmul
from tessera.backend.c.templates.reduction import schedule_reduce, schedule_rows
from tessera.target import Target


def make_templates(target: Target) -> dict[str, Template]:
    """The templates, specialized for ``target``'s ISA and thread count."""
    cpu = CpuTarget.of(target)
    table = {
        "conv2d": schedule_conv2d,
        "depthwise_conv2d": schedule_depthwise_conv2d,
        "matmul": schedule_matmul,
        "pool": schedule_pool,
        "reduce": schedule_reduce,
        "softmax": schedule_rows,
        "layernorm": schedule_rows,
        "injective": schedule_injective,
    }
    return {name: functools.partial(fn, target=cpu) for name, fn in table.items()}
