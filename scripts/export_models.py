"""Export the benchmark models to ONNX with their pretrained weights.

    uv run --group bench python scripts/export_models.py [resnet50 mobilenet_v2 bert gpt2]

(``make models`` exports all four.) For each model this writes, under
``data/models/`` (or ``$TESSERA_MODELS``):

- ``<name>.onnx``: opset 17, static shapes, batch 1;
- ``<name>.npz``: the fixed inputs, keyed by ONNX input name, and PyTorch's
  outputs for them, keyed by ONNX output name;
- an entry in ``manifest.json`` with the paths, the inputs and outputs, how
  to rebuild the PyTorch module offline (``torch``), the exporter used and
  the largest difference between ONNX Runtime and PyTorch on the inputs.

The inputs are realistic: a seeded random image normalized with the
ImageNet mean and standard deviation, and the tokenizer's ids for a fixed
English text (BERT: one paragraph padded to 128 tokens with its attention
mask; GPT-2: a longer text cut to 128 tokens). Every export is checked with
ONNX Runtime against PyTorch before anything is written, and models that
are already exported are skipped. This is the only place in the repository
that downloads anything: the weights, into the torchvision and Hugging Face
caches that the benchmarks later read offline.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnx

OPSET = 17
MAX_IR_VERSION = 9  # ONNX Runtime 1.31 loads up to 13; 9 is what the tests use too
TOKENS = 128
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

TEXT = (
    "The tensor compiler reads a neural network, rewrites its graph, groups "
    "operators into kernels and searches for the loop schedule that runs each "
    "kernel fastest on the machine in front of it."
)
LONG_TEXT = (
    "Apple silicon puts the CPU, the GPU and the memory on one package, so a "
    "model's weights never have to be copied to a separate device before a "
    "kernel can read them. The performance cores have wide vector units and, "
    "on the newest chips, a matrix extension that multiplies small tiles in "
    "a single instruction stream. The GPU runs thousands of threads grouped "
    "into threadgroups that share a fast scratchpad. Getting close to the "
    "peak of either one is mostly a matter of moving data well: choosing tile "
    "sizes that fit in registers and caches, keeping every core busy, and "
    "avoiding round trips to memory between operations that could have been "
    "fused. A compiler that measures candidate schedules on the hardware, "
    "and learns which ones are worth measuring, can find choices that a fixed "
    "heuristic misses, especially for the unusual shapes that real models "
    "contain. This paragraph exists only to give the language model a "
    "realistic prompt of more than one hundred and twenty eight tokens."
)


@dataclass
class Export:
    """A model to export: how to build it, its inputs, and its output names."""

    name: str
    torch: dict[str, str]
    inputs: Callable[[], dict[str, np.ndarray]]
    outputs: list[str]
    module: Callable[[], Any]
    tolerance: float = 1e-3  # max |onnxruntime - pytorch| / max |pytorch|
    notes: dict[str, Any] = field(default_factory=dict)


def image_input(seed: int = 0) -> dict[str, np.ndarray]:
    """A seeded random RGB image in [0, 1], normalized like ImageNet, NCHW."""
    rng = np.random.default_rng(seed)
    pixels = rng.random((224, 224, 3), dtype=np.float32)
    normalized = (pixels - IMAGENET_MEAN) / IMAGENET_STD
    return {"input": np.ascontiguousarray(normalized.transpose(2, 0, 1)[None], np.float32)}


def torchvision_module(name: str, weights: str) -> Callable[[], Any]:
    def build() -> Any:
        import torchvision

        enum = torchvision.models.get_model_weights(name)[weights]
        return torchvision.models.get_model(name, weights=enum).eval()

    return build


def bert_inputs() -> dict[str, np.ndarray]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    enc = tok(TEXT, padding="max_length", truncation=True, max_length=TOKENS, return_tensors="np")
    return {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }


def gpt2_inputs() -> dict[str, np.ndarray]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    ids = tok(LONG_TEXT, return_tensors="np")["input_ids"].astype(np.int64)
    if ids.shape[1] < TOKENS:
        raise RuntimeError(f"the GPT-2 prompt has only {ids.shape[1]} tokens, need {TOKENS}")
    return {"input_ids": ids[:, :TOKENS]}


def bert_module() -> Any:
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained("bert-base-uncased", attn_implementation="eager").eval()

    class LastHiddenState(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            # The default segment ids are zeros; passing them avoids the gather
            # transformers uses to slice its zero buffer (GatherElements in ONNX).
            segments = torch.zeros_like(input_ids)
            out = self.model(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segments
            )
            return out.last_hidden_state

    return LastHiddenState().eval()


def gpt2_module() -> Any:
    import torch
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
    model.config.use_cache = False

    class Logits(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model

        def forward(self, input_ids: Any) -> Any:
            # An explicit all-ones mask computes the same causal attention, and
            # skips transformers' packed-sequence check, whose torch.diff has no
            # ONNX export.
            mask = torch.ones_like(input_ids)
            return self.model(input_ids=input_ids, attention_mask=mask, use_cache=False).logits

    return Logits().eval()


EXPORTS = {
    "resnet50": Export(
        "resnet50",
        {"library": "torchvision", "name": "resnet50", "weights": "IMAGENET1K_V2"},
        image_input,
        ["logits"],
        torchvision_module("resnet50", "IMAGENET1K_V2"),
    ),
    "mobilenet_v2": Export(
        "mobilenet_v2",
        {"library": "torchvision", "name": "mobilenet_v2", "weights": "IMAGENET1K_V2"},
        image_input,
        ["logits"],
        torchvision_module("mobilenet_v2", "IMAGENET1K_V2"),
    ),
    "bert": Export(
        "bert",
        {
            "library": "transformers",
            "name": "bert-base-uncased",
            "class": "AutoModel",
            "output": "last_hidden_state",
            "exported_with": "attn_implementation=eager, token_type_ids=zeros, use_cache=False",
        },
        bert_inputs,
        ["last_hidden_state"],
        bert_module,
    ),
    "gpt2": Export(
        "gpt2",
        {
            "library": "transformers",
            "name": "gpt2",
            "class": "GPT2LMHeadModel",
            "output": "logits",
            "exported_with": "attn_implementation=eager, attention_mask=ones, use_cache=False",
        },
        gpt2_inputs,
        ["logits"],
        gpt2_module,
    ),
}


def export_onnx(module: Any, inputs: dict[str, np.ndarray], outputs: list[str], path: Path) -> str:
    """Export with the dynamo exporter when it is installed, else TorchScript's;
    returns the exporter's name."""
    import torch

    args = tuple(torch.from_numpy(a) for a in inputs.values())
    common: dict[str, Any] = {
        "input_names": list(inputs),
        "output_names": outputs,
        "opset_version": OPSET,
    }
    if importlib.util.find_spec("onnxscript") is not None:
        try:
            torch.onnx.export(module, args, str(path), dynamo=True, external_data=False, **common)
            return "dynamo"
        except Exception as e:  # fall back to the exporter that handles these models
            print(f"  dynamo exporter failed ({type(e).__name__}: {e}); using TorchScript")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the TorchScript exporter's deprecation notice
        torch.onnx.export(module, args, str(path), dynamo=False, do_constant_folding=True, **common)
    return "torchscript"


def check_with_onnxruntime(
    path: Path, inputs: dict[str, np.ndarray], expected: dict[str, np.ndarray]
) -> tuple[float, float]:
    """Max absolute and relative (to the largest magnitude) error of ONNX Runtime."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = dict(zip([o.name for o in session.get_outputs()], session.run(None, inputs), strict=True))
    worst_abs = worst_rel = 0.0
    for name, want in expected.items():
        diff = float(np.max(np.abs(got[name].astype(np.float64) - want)))
        worst_abs = max(worst_abs, diff)
        worst_rel = max(worst_rel, diff / float(np.max(np.abs(want))))
    return worst_abs, worst_rel


def export(spec: Export, directory: Path) -> dict[str, Any]:
    """Export one model; returns its manifest entry."""
    import torch

    onnx_path = directory / f"{spec.name}.onnx"
    npz_path = directory / f"{spec.name}.npz"
    tmp_path = directory / f".{spec.name}.onnx.tmp"
    started = time.monotonic()
    torch.manual_seed(0)
    inputs = spec.inputs()
    module = spec.module()
    with torch.inference_mode():
        result = module(*(torch.from_numpy(a) for a in inputs.values()))
    expected = {spec.outputs[0]: result.detach().float().numpy()}

    exporter = export_onnx(module, inputs, spec.outputs, tmp_path)
    model = onnx.load(str(tmp_path))
    if model.ir_version > MAX_IR_VERSION:
        model.ir_version = MAX_IR_VERSION
    onnx.checker.check_model(model)
    onnx.save(model, str(tmp_path))
    max_abs, max_rel = check_with_onnxruntime(tmp_path, inputs, expected)
    if max_rel > spec.tolerance:
        tmp_path.unlink()
        raise RuntimeError(
            f"{spec.name}: ONNX Runtime differs from PyTorch by {max_rel:.2e} (relative), "
            f"more than {spec.tolerance:.0e}"
        )
    os.replace(tmp_path, onnx_path)
    np.savez(npz_path, **inputs, **expected)
    print(
        f"  {spec.name}: {exporter} exporter, {onnx_path.stat().st_size / 1e6:.1f} MB, "
        f"onnxruntime vs pytorch max abs {max_abs:.2e}, max rel {max_rel:.2e}, "
        f"{time.monotonic() - started:.0f} s"
    )
    return {
        "onnx": onnx_path.name,
        "npz": npz_path.name,
        "inputs": [
            {"name": k, "shape": list(v.shape), "dtype": str(v.dtype)} for k, v in inputs.items()
        ],
        "outputs": [{"name": k, "shape": list(v.shape)} for k, v in expected.items()],
        "torch": spec.torch,
        "exporter": exporter,
        "opset": OPSET,
        "onnxruntime_max_abs_error": max_abs,
        "onnxruntime_max_rel_error": max_rel,
    }


def use_certificate_bundle() -> None:
    """Give ``urllib`` (torchvision's downloader) a certificate bundle.

    Some Python installations (python.org's on macOS) ship without one, and
    the download fails certificate verification. ``SSL_CERT_FILE`` is set only
    when it is unset: to certifi's bundle if certifi is installed, otherwise
    to the system's ``/etc/ssl/cert.pem`` if it exists.
    """
    if os.environ.get("SSL_CERT_FILE"):
        return
    try:
        import certifi

        os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        if Path("/etc/ssl/cert.pem").exists():
            os.environ["SSL_CERT_FILE"] = "/etc/ssl/cert.pem"


def main(argv: list[str]) -> int:
    use_certificate_bundle()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "names", nargs="*", metavar="NAME", help=f"{', '.join(EXPORTS)} (default: all)"
    )
    parser.add_argument("--force", action="store_true", help="export again even if present")
    args = parser.parse_args(argv)
    unknown = sorted(set(args.names) - set(EXPORTS))
    if unknown:
        parser.error(f"unknown models {unknown}; choose from {', '.join(EXPORTS)}")
    default_dir = Path(__file__).resolve().parents[1] / "data" / "models"
    directory = Path(os.environ.get("TESSERA_MODELS", default_dir))
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name in args.names or list(EXPORTS):
        entry = manifest.get(name)
        paths = [directory / f"{name}.onnx", directory / f"{name}.npz"]
        if entry is not None and all(p.exists() for p in paths) and not args.force:
            size = sum(p.stat().st_size for p in paths) / 1e6
            print(f"  {name}: already exported ({size:.1f} MB), skipped")
            continue
        manifest[name] = export(EXPORTS[name], directory)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    total = sum(p.stat().st_size for p in directory.glob("*.*") if p.is_file()) / 1e6
    print(f"{directory}: {total:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
