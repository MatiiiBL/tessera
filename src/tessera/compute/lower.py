"""Lowering of fused groups to unscheduled PrimFuncs (``compute.lower_group``)."""

from __future__ import annotations

from math import prod

from tessera import te
from tessera.compute.anchor import ANCHORS, ROLE, AnchorResult
from tessera.compute.elementwise import DEFINITIONS
from tessera.compute.tensor import TensorLike
from tessera.plan import FusedGroup, LoweredGroup
from tessera.tir.arith import Analyzer
from tessera.tir.functor import substitute_stmt
from tessera.tir.primitives.common import iter_paths, simplify_stmt, with_body
from tessera.tir.stmt import Block, Buffer, PrimFunc


def lower_group(group: FusedGroup) -> LoweredGroup:
    """The group as a PrimFunc: one parameter per group input (in order), then the
    output. Prologue operators are inlined into the loads of the main block (the
    anchor's), the epilogue operators form one block that reads the main block's
    result, and a group with no anchor is one block. Softmax and LayerNorm add
    their statistics blocks before the main block. Every block carries a
    ``role`` annotation: ``main``, ``epilogue``, ``injective`` or a statistic
    (``max``, ``sum``, ``mean``, ``var``)."""
    env: dict[int, TensorLike] = {}
    params: list[te.Tensor] = []
    for v in group.inputs:
        t = te.placeholder(v.shape, v.dtype, v.name)
        env[id(v)] = t
        params.append(t)
    anchor: AnchorResult | None = None
    for node in group.nodes:
        try:
            ins = [env[id(v)] for v in node.inputs]
        except KeyError:
            raise ValueError(
                f"group {group.name}: {node.name} reads a value from nowhere"
            ) from None
        if node is group.anchor:
            if node.op not in ANCHORS:
                raise ValueError(f"group {group.name}: {node.op} cannot anchor a kernel")
            anchor = ANCHORS[node.op](node, ins)
            env[id(node.output)] = anchor.value
        else:
            if node.op not in DEFINITIONS:
                raise ValueError(f"group {group.name}: no compute definition for {node.op}")
            env[id(node.output)] = DEFINITIONS[node.op](node, ins)
    out = env[id(group.output)]
    epilogue: str | None = None
    if anchor is not None and out is anchor.main:
        result = anchor.main
    else:
        role = "injective" if anchor is None else "epilogue"
        name = group.output.name
        if anchor is not None and name in {anchor.main.name, *(t.name for t in anchor.stages)}:
            name = f"{name}_epilogue"
        result = te.compute(
            out.shape,
            lambda *i: out[i],
            name,
            axis_names=[f"i{d}" for d in range(len(out.shape))],
            annotations={ROLE: role},
        )
        epilogue = None if anchor is None else result.name
    func = _unique_buffer_names(te.create_prim_func([*params, result], group.name))
    func = with_body(func, simplify_stmt(func.body, Analyzer()))
    if anchor is None:
        dims = {"SIZE": prod(group.output.shape)}
        return LoweredGroup(group, func, None, None, dims)
    roles = _roles(func)
    return LoweredGroup(
        group, func, roles["main"], roles.get("epilogue") if epilogue else None, anchor.dims
    )


def _roles(func: PrimFunc) -> dict[str, str]:
    """Block names by role annotation."""
    out: dict[str, str] = {}
    for path in iter_paths(func.body):
        node = path[-1]
        if isinstance(node, Block):
            role = dict(node.annotations).get(ROLE)
            if isinstance(role, str):
                out[role] = node.name
    return out


def _unique_buffer_names(func: PrimFunc) -> PrimFunc:
    """Rename allocated buffers whose names repeat another buffer's (code generators
    name variables after buffers)."""
    used = {p.name for p in func.params}
    renames: dict[Buffer, Buffer] = {}
    for path in iter_paths(func.body):
        node = path[-1]
        if not isinstance(node, Block):
            continue
        for b in node.alloc_buffers:
            name, k = b.name, 1
            while name in used:
                name = f"{b.name}_{k}"
                k += 1
            used.add(name)
            if name != b.name:
                renames[b] = Buffer(name, b.shape, b.dtype, b.scope)
    if not renames:
        return func
    return with_body(func, substitute_stmt(func.body, buffer_map=renames))
