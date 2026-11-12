"""The loop IR (TensorIR-style blocks and loops), its analyses and schedules."""

from tessera.tir import intrin, transform
from tessera.tir.analysis import structural_equal, structural_hash
from tessera.tir.arith import Analyzer, Interval, simplify
from tessera.tir.interpreter import interpret, python_source
from tessera.tir.printer import script
from tessera.tir.schedule import Schedule, ScheduleError
from tessera.tir.transform import compact_buffers, lower_blocks
from tessera.tir.verify import VerifyError, verify

__all__ = [
    "Analyzer",
    "Interval",
    "Schedule",
    "ScheduleError",
    "VerifyError",
    "compact_buffers",
    "interpret",
    "intrin",
    "lower_blocks",
    "python_source",
    "script",
    "simplify",
    "structural_equal",
    "structural_hash",
    "transform",
    "verify",
]
