"""Graph passes: canonicalization, folding, layout, fusion and memory planning."""

from tessera.passes.fusion import FusionOptions, fuse, verify_groups
from tessera.passes.memory import plan_memory
from tessera.passes.optimize import optimize

__all__ = ["FusionOptions", "fuse", "optimize", "plan_memory", "verify_groups"]
