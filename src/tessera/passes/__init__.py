"""Graph passes: canonicalization, folding, layout, fusion and memory planning."""

from tessera.passes.fusion import FusionOptions, fuse, verify_groups
from tessera.passes.memory import plan_memory

__all__ = ["FusionOptions", "fuse", "plan_memory", "verify_groups"]
