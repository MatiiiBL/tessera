# Evaluation

This document describes how Tessera is measured: the models, the systems
it is compared with, what is timed and how, the ablations, and how to
reproduce every number. The numbers themselves are in `results/`, written
by `tessera bench`; the tables below are copied from those files.

## Setup

- **Machine:** Apple M4, 4 performance and 6 efficiency cores, 10-core GPU,
  16 GB of unified memory, macOS. Each result file records the date, the
  commit, the OS and the load average when its run started.
- **Models:** ResNet-50 and MobileNetV2 (torchvision, `IMAGENET1K_V2`
  weights, 224×224), BERT-base (`bert-base-uncased`, sequence 128) and
  GPT-2 small (`gpt2`, a prefill of 128 tokens), exported to ONNX (opset
  17, static shapes, batch 1, float32) by `scripts/export_models.py`
  (`make models`). The script checks every export with ONNX Runtime against
  PyTorch before writing it; the largest relative differences are 1.0e-6
  (ResNet-50), 1.7e-6 (MobileNetV2), 2.2e-6 (BERT) and 1.2e-6 (GPT-2), and
  are recorded in `data/models/manifest.json`.
- **Inputs:** fixed and realistic: a seeded random image normalized with
  the ImageNet mean and standard deviation; for BERT, the tokenizer's ids
  of a fixed English sentence padded to 128 tokens with its attention mask;
  for GPT-2, a fixed paragraph cut to 128 tokens.

The language models are exported with eager attention
(`attn_implementation="eager"`: the softmax of a scaled `Q·Kᵀ` written out
as matrix products), which Tessera imports and fuses like any other graph.
BERT gets explicit zero segment ids and GPT-2 an explicit all-ones
attention mask; both are what the libraries use when the argument is
omitted, and passing them keeps two operators without an export rule out of
the graph.

## Systems

| system | what runs |
|---|---|
| `onnxruntime-cpu` | ONNX Runtime's CPU provider, all graph optimizations, default threads |
| `pytorch-cpu` | the PyTorch module, eager, `inference_mode`, default threads |
| `pytorch-mps` | the same module on the GPU through MPS |
| `tessera-<target>` | `tessera.compile` for `c:sme` (C with NEON and SME, 4 threads on the performance cores) or `metal`, with the committed tuning logs |
| `tessera-<target>-untuned` | the same with every template's default configuration |

The PyTorch baseline uses transformers' default attention (scaled dot
product attention), PyTorch's fastest eager path. It computes the same
function as the exported eager formulation, so the comparison does not
handicap PyTorch.

## Metrics

- **Latency:** wall-clock time of one inference with the inputs already in
  place (on the device for MPS; the call waits for the GPU to finish).
  Each system runs 10 untimed warmup inferences, then at least 50 timed
  ones; the tables give the median and the 10th and 90th percentiles. A
  row that cannot run 60 inferences within 60 seconds (only the
  unscheduled pipeline, below) gets as many as fit, at least 3, and says
  how many.
- **Agreement** with ONNX Runtime on the CPU, the reference: the largest
  absolute difference over every output element, the largest difference
  relative to the largest magnitude of the reference output, and, for the
  image models, the fraction of 16 inputs (the fixed image and 15 seeded
  random ones) on which both pick the same class.

## Ablations

Each ablation compiles the model again, untuned, with one part of the
pipeline changed:

- **Fusion off** (`fuse=False`): every operator is its own kernel. The
  tables report the number of kernels and the **bytes moved**, the sum
  over kernel launches of the bytes of every argument. That is what the
  kernels read and write if each touches its arguments once: a lower bound
  of memory traffic that does not depend on the hardware, and the quantity
  fusion reduces by keeping intermediates out of memory.
- **Graph rewrites off** (`optimize=False`): the minimal legal pipeline,
  canonical operators and the NHWC layout (constant weights are still
  relaid out at compile time) with dead code removed, but no constant
  folding beyond what import needs, BatchNorm decomposed into `Mul` and
  `Add` instead of folded into the convolution, no pattern rewrites
  (LayerNorm and GELU stay decomposed) and no common subexpression
  elimination. This measures the graph rewrites, not the layout.
- **No schedule** (`schedule=False`): no template is applied, so every loop
  is serial and nothing is vectorized, tiled or parallel. This is what the
  templates add.
- **Memory planning:** the arena size against the sum of the intermediate
  tensors it holds (every value a kernel writes into the arena). Values
  whose lifetimes do not overlap share bytes.
- **Tuning:** `tessera-<target>` against `tessera-<target>-untuned`, and
  random against model-guided search, which [tuning.md](tuning.md)
  describes and `results/tuning.md` reports.

## Reproducing

```sh
make models                                     # export the four models (downloads weights)
uv run --group bench tessera bench models       # results/models.md, models.json
uv run --group bench tessera bench ops          # results/ops.md: GEMM and ResNet-50 convolutions
uv run tessera bench tuning --real-tasks 2      # results/tuning.md: search strategies
uv run tessera tune data/models/resnet50.onnx -t c:sme --trials 1000   # extend a tuning log
```

`tessera bench models --model NAME` measures one model and keeps the
others' results, so a full run can be split into runs of a few minutes.
A run that starts while the one-minute load average is above 2 is marked
provisional in its file; final numbers are re-run on a quiet machine with
the same commands.

## Results

**Provisional (2026-11-12):** these were measured on a busy machine (load
averages 4 to 6.5 while each model ran) and before the backends' schedule
templates existed, so the Tessera row is the unscheduled pipeline only. The
full tables are in [results/models.md](../results/models.md).

Median latency in milliseconds, batch 1:

| model | ONNX Runtime CPU | PyTorch CPU | PyTorch MPS | Tessera c:sme, no schedule |
|---|---|---|---|---|
| ResNet-50 | 31.9 | 38.2 | 35.8 | 4,719 (12 runs) |
| MobileNetV2 | 4.8 | 36.9 | 23.5 | 183 |
| BERT-base | 47.1 | 63.4 | 21.1 | 17,130 (3 runs) |
| GPT-2 | 65.8 | 85.5 | 40.4 | 23,167 (3 runs) |

Every system, the unscheduled Tessera pipeline included, agrees with ONNX
Runtime to within 4e-6 of the output's largest magnitude; the baselines
pick the same top-1 class on all 16 images (the ablation rows compare the
fixed image only). Without schedules
every loop is serial, which is why that column is two to three orders of
magnitude slower: it is the baseline the templates are measured against.
On Metal the unscheduled pipeline runs each kernel on a single GPU thread;
one inference took 17 s for MobileNetV2 and 239 s for ResNet-50, too slow
to benchmark.

Memory planning, from the same runs: the arena holds 10.0 MiB for ResNet-50
(43.2 MiB of intermediate tensors), 6.5 MiB for MobileNetV2 (25.5 MiB),
2.3 MiB for BERT (72.1 MiB) and 3.4 MiB for GPT-2 (72.8 MiB).

## Limitations

- Batch 1 and float32 only; no int8 or float16 kernels.
- Latency is measured with the inputs already in place; copying inputs in
  and outputs out is not timed for any system.
- ONNX Runtime and PyTorch run with their default thread counts, which can
  include efficiency cores; Tessera's C target uses the 4 performance
  cores. The result files record each system's thread count.
