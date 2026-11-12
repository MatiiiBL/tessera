"""The systems Tessera is compared with: ONNX Runtime on the CPU and PyTorch eager
on the CPU and on the GPU (MPS).

Each baseline is a ``Runner``: ``outputs(inputs)`` runs once and returns
NumPy outputs by ONNX output name, and ``prepare(inputs)`` puts the inputs
where the system wants them (tensors on the device, feeds built) and
returns a call that runs the model once and waits for it to finish, which
is what gets timed. Threads are left at each library's default unless
``threads`` is given; the result files record the count each one used.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tessera.bench.zoo import ModelSpec, torch_module
from tessera.ir.graph import NDArray


@dataclass(frozen=True)
class Runner:
    """A way to run one model: ``outputs`` for correctness, ``prepare`` for timing."""

    name: str
    outputs: Callable[[Mapping[str, NDArray]], dict[str, NDArray]]
    prepare: Callable[[Mapping[str, NDArray]], Callable[[], object]]
    threads: int | None = None


def onnxruntime_cpu(path: str | Path, *, threads: int | None = None) -> Runner:
    """ONNX Runtime's CPU provider with every graph optimization enabled."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        options.intra_op_num_threads = threads
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    names = [o.name for o in session.get_outputs()]

    def outputs(inputs: Mapping[str, NDArray]) -> dict[str, NDArray]:
        return dict(zip(names, session.run(None, dict(inputs)), strict=True))

    def prepare(inputs: Mapping[str, NDArray]) -> Callable[[], object]:
        feeds = dict(inputs)
        return lambda: session.run(None, feeds)

    used = session.get_session_options().intra_op_num_threads
    return Runner("onnxruntime-cpu", outputs, prepare, threads=used or threads)


def torch_available(device: str) -> bool:
    """True if PyTorch is installed and ``device`` ("cpu" or "mps") can run."""
    try:
        import torch
    except ImportError:
        return False
    return device == "cpu" or (device == "mps" and bool(torch.backends.mps.is_available()))


def torch_eager(spec: ModelSpec, device: str = "cpu", *, threads: int | None = None) -> Runner:
    """The model's PyTorch module, run eagerly under ``inference_mode`` on ``device``."""
    import torch

    if threads:
        torch.set_num_threads(threads)
    module = torch_module(spec).to(device)
    input_names = [name for name, _, _ in spec.inputs]
    output_names = [name for name, _ in spec.outputs]
    keyword = spec.torch.get("library") == "transformers"

    def tensors(inputs: Mapping[str, NDArray]) -> dict[str, Any]:
        return {
            k: torch.from_numpy(np.ascontiguousarray(inputs[k])).to(device) for k in input_names
        }

    def call(feeds: dict[str, Any]) -> Any:
        with torch.inference_mode():
            return module(**feeds) if keyword else module(*feeds.values())

    def sync() -> None:
        if device == "mps":
            torch.mps.synchronize()

    def outputs(inputs: Mapping[str, NDArray]) -> dict[str, NDArray]:
        result = call(tensors(inputs))
        sync()
        if isinstance(result, torch.Tensor):
            values = [result]
        else:
            values = [result[name] for name in output_names]
        return {
            name: v.detach().to("cpu").float().numpy()
            for name, v in zip(output_names, values, strict=True)
        }

    def prepare(inputs: Mapping[str, NDArray]) -> Callable[[], object]:
        feeds = tensors(inputs)

        def run() -> None:
            call(feeds)
            sync()

        return run

    return Runner(f"pytorch-{device}", outputs, prepare, threads=torch.get_num_threads())
