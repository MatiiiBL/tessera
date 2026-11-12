"""The graph interpreter: runs a graph node by node with the NumPy references.

It is the oracle compiled code is tested against. ``float64=True`` computes
every float value in float64 instead of the graph's float32, which gives a
reference whose own rounding error is negligible next to a kernel's.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from tessera.ir.graph import Graph, NDArray, TensorType, Value


def interpret(
    graph: Graph,
    inputs: Mapping[str, NDArray],
    *,
    float64: bool = False,
    intermediates: bool = False,
) -> dict[str, NDArray]:
    """Run ``graph`` on ``inputs`` (by input name) and return its outputs by name.

    With ``float64``, float inputs and constants are converted to float64 and
    every float value stays float64 (an explicit ``Cast`` still rounds to its
    type first). With ``intermediates``, every value computed is returned, not
    only the outputs. Each node's results are checked against its output types.
    """
    env: dict[int, NDArray] = {}
    _bind_inputs(graph, inputs, env, float64)
    keep = {id(v) for v in graph.outputs}
    last_use: dict[int, int] = {}
    for index, node in enumerate(graph.nodes):
        for v in node.inputs:
            last_use[id(v)] = index
    results: dict[str, NDArray] = {}

    for index, node in enumerate(graph.nodes):
        args = [_lookup(v, env, float64) for v in node.inputs]
        try:
            with np.errstate(all="ignore"):
                outs = node.opdef.reference(args, node.attrs)
        except Exception as e:
            e.add_note(f"while interpreting {node!r}")
            raise
        if len(outs) != len(node.outputs):
            raise RuntimeError(f"{node!r}: reference returned {len(outs)} results")
        for value, out in zip(node.outputs, outs, strict=True):
            array = _check(value, np.asarray(out), float64, node_text=repr(node))
            env[id(value)] = array
            if intermediates:
                results[value.name] = array
        if not intermediates:
            for v in node.inputs:
                if last_use.get(id(v)) == index and id(v) not in keep and v.producer is not None:
                    env.pop(id(v), None)

    for v in graph.outputs:
        results[v.name] = np.array(_lookup(v, env, float64), copy=True)
    return results


def random_inputs(graph: Graph, seed: int = 0, int_high: int = 2) -> dict[str, NDArray]:
    """Random arrays for every graph input: floats from N(0, 1), ints in [0, int_high)."""
    rng = np.random.default_rng(seed)
    out: dict[str, NDArray] = {}
    for v in graph.inputs:
        dtype = v.dtype.numpy
        if v.dtype.is_float:
            out[v.name] = rng.standard_normal(v.shape).astype(dtype)
        elif v.dtype.is_int:
            out[v.name] = rng.integers(0, int_high, v.shape).astype(dtype)
        else:
            out[v.name] = rng.integers(0, 2, v.shape).astype(dtype)
    return out


def _bind_inputs(
    graph: Graph, inputs: Mapping[str, NDArray], env: dict[int, NDArray], float64: bool
) -> None:
    names = {v.name for v in graph.inputs}
    unknown = sorted(set(inputs) - names)
    if unknown:
        raise KeyError(f"graph {graph.name} has no inputs {unknown}; inputs are {sorted(names)}")
    for v in graph.inputs:
        if v.name not in inputs:
            raise KeyError(f"missing input {v.name!r} ({v.type})")
        array = np.asarray(inputs[v.name])
        if tuple(array.shape) != v.shape:
            raise ValueError(f"input {v.name!r} has shape {array.shape}, expected {v.type}")
        if array.dtype != v.dtype.numpy and not (float64 and v.dtype.is_float):
            raise ValueError(f"input {v.name!r} has type {array.dtype}, expected {v.type}")
        env[id(v)] = _widen(array, v.type, float64)


def _lookup(v: Value, env: dict[int, NDArray], float64: bool) -> NDArray:
    array = env.get(id(v))
    if array is not None:
        return array
    if v.data is not None:
        array = _widen(v.data, v.type, float64)
        env[id(v)] = array
        return array
    raise KeyError(f"value {v!r} is read before it is computed")


def _widen(array: NDArray, type: TensorType, float64: bool) -> NDArray:
    if float64 and type.dtype.is_float:
        return array.astype(np.float64)
    return array


def _check(value: Value, array: NDArray, float64: bool, node_text: str) -> NDArray:
    if tuple(array.shape) != value.shape:
        raise RuntimeError(f"{node_text}: reference gave shape {array.shape} for {value!r}")
    if float64 and value.dtype.is_float:
        return array if array.dtype == np.float64 else array.astype(np.float64)
    if array.dtype != value.dtype.numpy:
        raise RuntimeError(f"{node_text}: reference gave type {array.dtype} for {value!r}")
    return array
