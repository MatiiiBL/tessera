"""Lower every group of a graph and compare it with the operators' references."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from tessera import ir, passes, tir
from tessera.compute import lower_group
from tessera.ir import Graph, Node
from tessera.plan import FusedGroup, LoweredGroup

type NDArray = npt.NDArray[Any]


def group_reference(group: FusedGroup, values: dict[str, NDArray]) -> NDArray:
    """The group's output computed by the operator references in float64."""
    env: dict[int, NDArray] = {}
    for v in group.inputs:
        data = values[v.name]
        env[id(v)] = data.astype(np.float64) if data.dtype.kind == "f" else data
    for node in group.nodes:
        args = [env[id(v)] for v in node.inputs]
        with np.errstate(all="ignore"):
            (out,) = node.opdef.reference(args, node.attrs)
        env[id(node.output)] = np.asarray(out)
    return env[id(group.output)]


def values_of(graph: Graph, inputs: dict[str, NDArray]) -> dict[str, NDArray]:
    values = dict(inputs)
    values.update(ir.interpret(graph, inputs, intermediates=True))
    for v in graph.constants():
        assert v.data is not None
        values[v.name] = v.data
    return values


def check_group(
    group: FusedGroup, values: dict[str, NDArray], rtol: float = 2e-4, *, pad_block: bool = False
) -> LoweredGroup:
    lowered = lower_group(group, pad_block=pad_block)
    tir.verify(lowered.func)
    args = [np.array(values[v.name], order="C") for v in group.inputs]
    dtype = group.output.dtype.numpy
    out = np.full(group.output.shape, np.nan if dtype.kind == "f" else 0, dtype)
    tir.interpret(lowered.func, [*args, out], check_bounds=True)
    expected = group_reference(group, values)
    if out.dtype.kind == "f":
        scale = max(1.0, float(np.nanmax(np.abs(expected), initial=0.0)))
        np.testing.assert_allclose(out, expected, rtol=rtol, atol=rtol * scale, err_msg=group.name)
    else:
        np.testing.assert_array_equal(out, expected.astype(out.dtype), err_msg=group.name)
    return lowered


def check_graph(
    graph: Graph,
    inputs: dict[str, NDArray] | None = None,
    *,
    fuse: bool = True,
    rtol: float = 2e-4,
    pad_block: bool = False,
) -> list[LoweredGroup]:
    """Fuse ``graph``, lower every group that is not a view and compare each one."""
    graph.verify()
    inputs = inputs if inputs is not None else ir.random_inputs(graph, seed=1)
    values = values_of(graph, inputs)
    groups = passes.fuse(graph, enabled=fuse)
    passes.verify_groups(graph, groups)
    return [check_group(g, values, rtol, pad_block=pad_block) for g in groups if not g.is_view]


def single_group(node: Node) -> FusedGroup:
    """A group holding just ``node`` (inputs: everything it reads)."""
    from tessera.passes.fusion import pattern_of

    anchor = node if node.opdef.kind >= ir.OpKind.REDUCTION else None
    inputs: list[ir.Value] = []
    for v in node.inputs:
        if v not in inputs:
            inputs.append(v)
    return FusedGroup(node.name, [node], inputs, node.output, anchor, pattern_of(anchor))
