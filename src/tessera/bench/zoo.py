"""The benchmark models exported by ``scripts/export_models.py``.

``make models`` writes, under ``data/models/`` (or ``$TESSERA_MODELS``),
``<name>.onnx``, ``<name>.npz`` (the fixed inputs keyed by ONNX input name
and PyTorch's outputs keyed by output name) and ``manifest.json``, which
describes each model and how to rebuild its PyTorch module offline: the
weights are already in the torchvision and Hugging Face caches, so nothing
here touches the network.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tessera.bench.report import repo_root
from tessera.ir.graph import NDArray

MODELS = ("resnet50", "mobilenet_v2", "bert", "gpt2")
CLASSIFIERS = frozenset({"resnet50", "mobilenet_v2"})


class MissingModels(FileNotFoundError):
    """The exported models (or their cached weights) are not there."""


def models_dir() -> Path:
    return Path(os.environ.get("TESSERA_MODELS", repo_root() / "data" / "models"))


@dataclass(frozen=True)
class ModelSpec:
    """One exported model, as described by the manifest."""

    name: str
    onnx: Path
    npz: Path
    inputs: tuple[tuple[str, tuple[int, ...], str], ...]  # name, shape, dtype
    outputs: tuple[tuple[str, tuple[int, ...]], ...]
    torch: dict[str, str]

    @property
    def classifier(self) -> bool:
        return self.name in CLASSIFIERS

    @property
    def input_shapes(self) -> dict[str, tuple[int, ...]]:
        return {name: shape for name, shape, _ in self.inputs}

    def sample_inputs(self) -> dict[str, NDArray]:
        """The fixed inputs saved at export time."""
        with np.load(self._existing(self.npz)) as data:
            return {name: data[name] for name, _, _ in self.inputs}

    def torch_outputs(self) -> dict[str, NDArray]:
        """PyTorch's outputs for ``sample_inputs``, saved at export time."""
        with np.load(self._existing(self.npz)) as data:
            return {name: data[name] for name, _ in self.outputs if name in data}

    def random_inputs(self, seed: int) -> dict[str, NDArray]:
        """Inputs of the right shapes: images ~ N(0, 1), token ids in a safe range."""
        rng = np.random.default_rng(seed)
        out: dict[str, NDArray] = {}
        for name, shape, dtype in self.inputs:
            if dtype.startswith("int"):
                if "mask" in name:
                    out[name] = np.ones(shape, dtype=dtype)
                else:
                    out[name] = rng.integers(0, 20000, size=shape, dtype=dtype)
            else:
                out[name] = rng.standard_normal(shape).astype(dtype)
        return out

    def onnx_path(self) -> Path:
        return self._existing(self.onnx)

    def _existing(self, path: Path) -> Path:
        if not path.exists():
            raise MissingModels(f"{path} is missing: run `make models` first")
        return path


def load_manifest(directory: Path | None = None) -> dict[str, ModelSpec]:
    """Every model in ``manifest.json``, by name."""
    root = directory if directory is not None else models_dir()
    path = root / "manifest.json"
    if not path.exists():
        raise MissingModels(f"{path} is missing: run `make models` first")
    raw = json.loads(path.read_text())
    entries = raw["models"] if isinstance(raw, dict) and "models" in raw else raw
    if isinstance(entries, dict):
        entries = [{"name": k, **v} for k, v in entries.items()]
    specs: dict[str, ModelSpec] = {}
    for e in entries:
        specs[e["name"]] = ModelSpec(
            name=e["name"],
            onnx=_resolve(root, e["onnx"]),
            npz=_resolve(root, e["npz"]),
            inputs=tuple(_io(i) for i in e["inputs"]),
            outputs=tuple(_io(o)[:2] for o in e["outputs"]),
            torch=dict(e.get("torch", {})),
        )
    return specs


def _resolve(root: Path, path: str) -> Path:
    """Manifest paths are relative to the manifest or to the checkout."""
    p = Path(path)
    if p.is_absolute():
        return p
    for candidate in (root / p, repo_root() / p, root / p.name):
        if candidate.exists():
            return candidate
    return root / p.name


def _io(entry: Any) -> tuple[str, tuple[int, ...], str]:
    """An input or output, as ``{"name", "shape", "dtype"}`` or ``[name, shape, dtype]``."""
    if isinstance(entry, dict):
        return str(entry["name"]), tuple(entry["shape"]), str(entry.get("dtype", "float32"))
    name, shape, *rest = entry
    return str(name), tuple(shape), str(rest[0]) if rest else "float32"


def torch_module(spec: ModelSpec) -> Any:
    """The model as an eager PyTorch module in eval mode, from cached weights only."""
    library = spec.torch.get("library")
    if library == "torchvision":
        return _torchvision(spec)
    if library == "transformers":
        return _transformers(spec)
    raise ValueError(f"model {spec.name}: unknown torch library {library!r}")


def _torchvision(spec: ModelSpec) -> Any:
    import torch
    import torchvision

    name = spec.torch["name"]
    weights_name = spec.torch["weights"]
    if "." in weights_name:
        weights = torchvision.models.get_weight(weights_name)
    else:
        weights = torchvision.models.get_model_weights(name)[weights_name]
    cached = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    if not cached.exists():
        raise MissingModels(f"{cached} is not cached: run `make models` first")
    module = torchvision.models.get_model(name, weights=weights)
    return module.eval()


def _transformers(spec: ModelSpec) -> Any:
    import transformers

    cls = getattr(transformers, spec.torch.get("class", "AutoModel"))
    try:
        module = cls.from_pretrained(spec.torch["name"], local_files_only=True)
    except OSError as e:
        raise MissingModels(f"{spec.torch['name']} is not cached: run `make models` first") from e
    return module.eval()
