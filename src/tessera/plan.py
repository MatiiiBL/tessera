"""The records that cross package boundaries in the compilation pipeline.

    Graph --passes.fuse--> [FusedGroup] --compute.lower_group--> [LoweredGroup]
          --template + Config--> [PrimFunc] --backend.build--> CompiledModel
    Graph + [FusedGroup] --passes.plan_memory--> MemoryPlan
    [PrimFunc] + MemoryPlan --driver--> ExecutionPlan

See docs/design.md for who produces and who consumes each one.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field

from tessera.ir.graph import NDArray, Node, TensorType, Value
from tessera.tir.stmt import PrimFunc

# Template keys (FusedGroup.pattern). Each backend registers one template per key.
PATTERNS = (
    "conv2d",  # Conv2d, groups == 1, with elementwise epilogue
    "depthwise_conv2d",  # Conv2d, groups == channels
    "matmul",  # MatMul (batched or not), with prologue and epilogue
    "pool",  # MaxPool2d / AvgPool2d
    "reduce",  # ReduceSum / ReduceMean / ReduceMax
    "softmax",
    "layernorm",
    "injective",  # only elementwise / broadcast / injective ops
)


@dataclass
class FusedGroup:
    """Nodes compiled into one kernel.

    Invariants (checked by ``passes.verify_groups``): ``nodes`` are in
    topological order; the group has exactly one output; no value defined
    inside the group other than ``output`` is read outside it; at most one node
    is of kind REDUCTION or ANCHOR (``anchor``), and nodes before it in the
    group are at most INJECTIVE (prologue), nodes after it at most BROADCAST
    (epilogue).
    """

    name: str  # unique, a valid C identifier; becomes the PrimFunc name
    nodes: list[Node]
    inputs: list[Value]  # values read from outside, in parameter order
    output: Value
    anchor: Node | None
    pattern: str  # one of PATTERNS

    @property
    def is_view(self) -> bool:
        """A group of only Reshape nodes: no kernel, the output aliases the input."""
        return all(n.op == "Reshape" for n in self.nodes)


@dataclass(frozen=True)
class LoweredGroup:
    """A group as an unscheduled PrimFunc, ready for a template.

    ``func.params`` are the group's inputs in ``FusedGroup.inputs`` order,
    then the output. Prologue ops are inlined into the main block's loads;
    the epilogue ops are one block that reads the main block's result.
    ``dims`` gives a template the problem sizes by name (conv: N H W C KH KW
    O OH OW SH SW; matmul: B M N K; reductions: OUTER R INNER).
    """

    group: FusedGroup
    func: PrimFunc
    main_block: str | None  # the anchor's block (None for injective groups)
    epilogue_block: str | None
    dims: Mapping[str, int] = field(default_factory=dict)

    @property
    def pattern(self) -> str:
        return self.group.pattern


class StorageKind(enum.StrEnum):
    INPUT = "input"  # a model input, passed to run()
    OUTPUT = "output"  # a model output, returned by run()
    CONSTANT = "constant"  # inside the constant blob (weights)
    ARENA = "arena"  # inside the scratch arena (intermediate values)


@dataclass(frozen=True)
class BufferRef:
    """Where a value lives at run time."""

    kind: StorageKind
    offset: int  # bytes into the arena or the constant blob; 0 for inputs/outputs
    nbytes: int
    index: int = -1  # position among the model's inputs or outputs


@dataclass
class MemoryPlan:
    """Storage for every value a kernel reads or writes, by value name.

    Arena offsets come from liveness over the group order: two values share
    bytes only if their lifetimes do not overlap. Offsets are multiples of
    ``alignment``.
    """

    refs: dict[str, BufferRef]
    arena_bytes: int
    constants: NDArray  # uint8 blob holding every constant at its CONSTANT offset
    alignment: int = 256


@dataclass(frozen=True)
class KernelLaunch:
    """One kernel call: ``args`` line up with the PrimFunc's params."""

    kernel: str
    args: tuple[BufferRef, ...]


@dataclass
class ExecutionPlan:
    """Everything a backend needs to run a model, besides the kernels' code."""

    name: str
    launches: list[KernelLaunch]
    memory: MemoryPlan
    inputs: list[tuple[str, TensorType]]
    outputs: list[tuple[str, TensorType]]
    funcs: dict[str, PrimFunc] = field(default_factory=dict)
