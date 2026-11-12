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
  ones; the tables give the median and the 10th and 90th percentiles.
  Tessera's CPU rows pool their 50 runs from 5 fresh processes of 10 runs
  each (each after its own warmup): macOS sometimes keeps one of the C
  runtime's SME workers on the efficiency cluster for a whole process, which
  then runs with one thread fewer (docs/cpu.md, Limitations), so a single
  process's runs would all share that placement. The kernel benchmarks on
  the CPU are pooled the same way. A row that cannot run 60 inferences
  within 60 seconds (only the unscheduled pipeline, below) gets as many as
  fit in one process, at least 3, and says how many.
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
make models   # export the four models (the only step that downloads anything)
uv run --group bench tessera bench models --target c:sme --target c:neon --target metal \
    --ablation-target c:sme --ablation-target metal --final     # results/models.{md,json}
uv run --group bench tessera bench ops --target c:sme --target metal --final   # results/ops.*
uv run tessera bench tuning --target c:sme --final               # results/tuning.*
```

The tuning logs these use are committed; [tuning.md](tuning.md) gives the
commands that made them. `tessera bench models --model NAME` measures one
model and keeps the others' results, so a run can be split. Without
`--final` or `--provisional`, a run that starts while the one-minute load
average is above 2 is marked provisional in its file. The final numbers
were measured with nothing else running and are marked final; the load
averages in the files are mostly the benchmarks themselves.

## Results

Measured 2026-11-12 between 13:43 and 13:53 UTC with nothing else
running (one-minute load averages 1.4 to 2.5, most of it the benchmark
itself), at commit `e047e7f`, with the tuning logs committed in
`results/tuning/` ([tuning.md](tuning.md) describes how they were made).
The full tables, with the 10th and 90th percentiles and every ablation,
are in [results/models.md](../results/models.md).

Median latency in milliseconds, batch 1:

| model | ONNX Runtime CPU | PyTorch CPU | PyTorch MPS | Tessera c:sme | Tessera Metal |
|---|---|---|---|---|---|
| ResNet-50 | 12.14 | 13.18 | 12.28 | 11.52 | **10.71** |
| MobileNetV2 | 2.43 | 15.25 | 6.62 | 2.15 | **1.74** |
| BERT-base | 21.97 | 26.46 | 14.85 | 18.06 | **10.31** |
| GPT-2 | 30.54 | 39.10 | 22.13 | 32.68 | **14.18** |

Tessera on the GPU is the fastest system on all four models: 1.13 times
ONNX Runtime on ResNet-50, 1.40 times ONNX Runtime on MobileNetV2, 1.44
times PyTorch MPS on BERT and 1.56 times PyTorch MPS on GPT-2 (each against
the fastest baseline for that model). On the CPU, Tessera's `c:sme` code is
faster than ONNX Runtime on ResNet-50 (1.05 times), MobileNetV2 (1.13) and
BERT (1.22), and slower on GPT-2 (0.93). Every Tessera output agrees with
ONNX Runtime to within 3e-6 of the output's largest magnitude, and every
system picks the same top-1 class as ONNX Runtime on all 16 images.

ResNet-50 on `c:sme` is the one place where the runtime's thread placement
shows (see Metrics): the tuned row's runs spread from 11.17 to 15.05 ms
(10th to 90th percentile) and the untuned row's from 11.80 to 16.34 ms,
because some of the five processes ran with one SME worker on the
efficiency cluster. For the same reason the untuned "full" row of the
ablation table, which compiles exactly what the untuned row compiles, has
a median of 15.49 ms; the `c:sme` ablation differences for ResNet-50 are
within this placement noise.

**Tuning.** Every kernel takes its configuration from the logs (the tables
count these as "tuned" even where validation chose the template's
default). Tuned against untuned, median latency:

| model | c:sme | Metal |
|---|---|---|
| ResNet-50 | 12.02 to 11.52 (1.04 times) | 16.92 to 10.71 (1.58 times) |
| MobileNetV2 | 3.15 to 2.15 (1.47 times) | 1.95 to 1.74 (1.12 times) |
| BERT-base | 21.98 to 18.06 (1.22 times) | 10.54 to 10.31 (1.02 times) |
| GPT-2 | 32.36 to 32.68 (0.99 times) | 14.65 to 14.18 (1.03 times) |

GPT-2 on `c:sme` is unchanged within noise. Without the SME microkernels
(`c:neon`, untuned) the same models take 26.03, 3.01, 60.79 and 94.43 ms:
2.2, 0.95, 2.8 and 2.9 times the untuned `c:sme` code.

**Ablations** (untuned, `c:sme` and Metal; [results/models.md](../results/models.md)):

- *Fusion* cuts the kernel count to 37 to 53% (ResNet-50 56 against 123,
  MobileNetV2 54 against 101, BERT 134 against 320, GPT-2 111 against 304)
  and the bytes moved by 23 to 43% (206 against 322 MiB, 66 against 115,
  578 against 787, 813 against 1,061). On Metal it makes every model
  faster: 1.02, 1.19, 1.09 and 1.06 times. On `c:sme` the effect is within
  noise (1.04 times on MobileNetV2, 1.02 on BERT, 1.00 on GPT-2; ResNet-50
  is within the placement noise above).
- *Graph rewrites* matter for the transformers, whose LayerNorm and GELU
  otherwise stay decomposed into many small operators: BERT 1.13 times
  faster with them on Metal and 1.04 on `c:sme`, GPT-2 1.08 and 1.22. The
  CNNs have the same kernels with and without them (a BatchNorm left as
  `Mul` and `Add` fuses into the convolution's epilogue), and their times
  are equal within noise.
- *No schedule:* without templates every loop is serial, and on `c:sme`
  one inference takes 2,937 ms for ResNet-50 (20 runs), 89 ms for
  MobileNetV2, 9,173 ms for BERT (6 runs) and 11,216 ms for GPT-2 (5 runs):
  28 to 417 times the untuned scheduled code. Unscheduled Metal kernels run
  on one GPU thread each; one inference took 17 s for MobileNetV2 and
  239 s for ResNet-50 in an earlier run, too slow to benchmark.
- *Memory planning:* the arena holds 10.0 MiB for ResNet-50 against 43.2 MiB
  of intermediate tensors, 6.5 against 25.5 for MobileNetV2, 2.3 against
  72.1 for BERT and 3.4 against 72.8 for GPT-2: 3.9 to 31 times less
  memory than giving every intermediate its own buffer. Without fusion
  there are more intermediates (101.5, 50.2, 176.7 and 190.1 MiB) and the
  arena grows to 11.5, 10.0, 3.8 and 3.8 MiB.

### Kernels

From [results/ops.md](../results/ops.md) (measured 13:53 UTC, same
session). Float32 GEMM `C = A @ B` with `B` constant, in GFLOP/s, against
NumPy (which calls Accelerate) and PyTorch on MPS:

| M×N×K | NumPy | PyTorch MPS | Tessera c:sme | Tessera Metal |
|---|---|---|---|---|
| 256³ | 1,379 | 113 | 1,069 | 143 |
| 1024³ | 1,871 | 2,041 | 1,643 | 1,588 |
| 2048³ | 1,898 | 2,747 | 1,348 | 2,673 |
| 128×768×768 | 1,102 | 477 | 1,618 | 536 |
| 128×3072×768 | 1,282 | 1,196 | 1,734 | 1,274 |
| 128×768×3072 | 1,312 | 1,170 | 1,520 | 1,243 |
| 128×50257×768 | 1,360 | 1,332 | 894 | 2,325 |

On the shapes of the transformer layers (128 rows), Tessera's `c:sme`
kernels are 1.16 to 1.47 times Accelerate; on large square matrices and
GPT-2's vocabulary projection they are slower (0.71 times at 2048³, 0.66
on the projection). Tessera's Metal kernels are within 3% of MPS at 2048³
and 1.75 times MPS on the vocabulary projection; both GPU libraries pay a
fixed cost per call that dominates small products.

ResNet-50's 23 distinct convolution shapes, each timed inside the compiled
model with its fused epilogue (Tessera) or alone (PyTorch): Tessera on Metal
is faster than PyTorch MPS on 19 of them and than PyTorch on the CPU on 16
(median 1,374 GFLOP/s against 346 and 1,183); Tessera on `c:sme` is faster
than PyTorch on the CPU on 4 (median 784 GFLOP/s). On Metal, six of the
seven strided shapes are the slow ones (212 to 322 GFLOP/s).

## Limitations

- Batch 1 and float32 only; no int8 or float16 kernels.
- Latency is measured with the inputs already in place; copying inputs in
  and outputs out is not timed for any system.
- ONNX Runtime and PyTorch run with their default thread counts (PyTorch
  uses 4 here); Tessera's C target uses the 4 performance cores. The
  result files record each system's thread count.
- On `c:sme`, the runtime's thread placement is decided per process and
  sometimes leaves a worker on an efficiency core for the whole process;
  pooling five processes makes the medians representative but leaves wide
  percentiles where it happens (ResNet-50 above).
- The tuning logs were made on this M4: their records name the target,
  the thread count and the CPU (`c-sme-t4-Apple_M4`), so a machine with a
  different CPU or thread count compiles with the template defaults.
