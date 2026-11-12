# Tessera design

Tessera compiles neural networks for Apple silicon. It imports an ONNX model
with static input shapes, optimizes the graph, fuses operators into kernels,
lowers each kernel to a loop IR, schedules the loops with templates whose
knobs are autotuned, and generates either C (with NEON and SME microkernels
on the CPU) or Metal Shading Language (on the GPU). The result is a
`CompiledModel` that runs the whole network from one native call (C) or one
command buffer (Metal).

This document is the contract between packages: the records they exchange,
the functions each one provides, the operator set and the conventions every
package follows. Each package has its own document with its internals and
measurements.

## 1. Pipeline

```
model.onnx
  │ frontend.onnx.load(path, input_shapes)          → Graph (ONNX semantics, NCHW)
  │ passes.optimize(graph)                          → Graph (canonical ops, NHWC, folded)
  │ passes.fuse(graph)                              → [FusedGroup]
  │ passes.plan_memory(graph, groups)               → MemoryPlan
  │ compute.lower_group(group)                      → LoweredGroup (unscheduled PrimFunc)
  │ backend.templates()[pattern](sch, lowered, cfg) → scheduled PrimFunc
  │      cfg = tuning log best, or the template's defaults
  │ driver builds ExecutionPlan(launches, memory, funcs)
  │ backend.build(plan, target)                     → CompiledModel
  ▼
compiled.run({"input": array}) → {"output": array}
```

The reference semantics of every operator is its NumPy implementation in the
operator registry. `ir.interpret(graph, inputs)` runs a graph with them and
is the oracle every compiled result is tested against.

## 2. Packages and ownership

| Package | Contents | Document |
|---|---|---|
| `tessera.ir` | Graph IR, operator registry, operator definitions, interpreter | graph.md |
| `tessera.frontend` | ONNX import | graph.md |
| `tessera.passes` | canonicalization, folding, layout, fusion, memory planning | graph.md |
| `tessera.tir` | loop IR, printer, simplifier, bounds, interpreter, schedule primitives, tensorize intrinsics | tir.md |
| `tessera.te` | tensor expressions → PrimFunc | tir.md |
| `tessera.compute` | compute definition of every operator; `lower_group` | tir.md |
| `tessera.backend.c` | C codegen, clang build cache, thread pool, executor, CPU templates, NEON/SME microkernels | cpu.md |
| `tessera.backend.metal` | MSL codegen, Objective-C runtime shim, executor, GPU templates | metal.md |
| `tessera.tune` | knob API, task extraction, measurement, search, cost model, tuning logs | tuning.md |
| `tessera.driver`, `tessera.cli`, `tessera.bench` | `compile()`, the `tessera` command, benchmarks and baselines | evaluation.md |

Shared contract modules, changed only by agreement: `tessera/dtype.py`,
`tessera/target.py`, `tessera/plan.py`, `tessera/backend/base.py`,
`tessera/tir/expr.py`, `tessera/tir/stmt.py`, the signatures in
`tessera/tir/schedule.py`, `tessera/tune/space.py`, `tessera/ir/graph.py`
and `tessera/ir/ops.py`. Adding to them (a helper, a new node kind) is fine
when existing code keeps working; changing or removing anything is agreed
first.

## 3. Operators

These are the operators of the graph IR after `passes.optimize`. The
frontend may create others while importing (Section 4), but they must be
gone after optimization. Attributes are normalized: axes non-negative, one
entry per spatial dimension, all shapes static.

**Elementwise** (`ELEMWISE`, one input unless noted, output like input):
`Relu`, `Sigmoid`, `Tanh`, `Exp`, `Log`, `Sqrt`, `Rsqrt` (1/√x), `Erf`,
`Neg`, `Abs`, `Silu` (x·σ(x)), `HardSwish`, `HardSigmoid(alpha, beta)`,
`Gelu(approximate: "none" | "tanh")`, `Clip(min: float, max: float)`,
`Cast(to: str)` (a DType value).

**Broadcast** (`BROADCAST`, NumPy broadcasting): `Add`, `Sub`, `Mul`, `Div`,
`Pow`, `Maximum`, `Minimum`; `Equal`, `Less`, `Greater` (bool output);
`Where(cond, a, b)`; `BatchNorm(epsilon)` with inputs (x, scale, bias, mean,
var) on channel axis 1, inference only (passes fold it into the preceding
`Conv2d` or decompose it into `Mul` + `Add`).

**Injective** (`INJECTIVE`): `Reshape(shape)` (no -1 or 0 left);
`Transpose(perm)`; `Slice(starts, ends, steps)` (one entry per axis, steps
≥ 1, clamped); `Concat(axis)`; `Gather(axis)` with inputs (data, indices),
output shape `data.shape[:axis] + indices.shape + data.shape[axis+1:]`;
`Expand(shape)`; `Pad(pads, value)` constant padding, `pads` as ONNX
(all begins, then all ends).

**Reduction** (`REDUCTION`): `ReduceSum`, `ReduceMean`, `ReduceMax`
(`axes`, `keepdims` 0/1); `Softmax(axis)`; `LayerNorm(axis, epsilon)` with
inputs (x, scale, bias), normalizing over axes `axis..rank-1`;
`MaxPool2d` and `AvgPool2d` (`kernel` (kh, kw), `strides`, `pads`
(top, left, bottom, right), `layout` "NCHW" | "NHWC", AvgPool also
`count_include_pad`).

**Anchor** (`ANCHOR`): `Conv2d(strides, pads, dilations, groups, layout)` with
inputs (x, w[, bias]); the weight is OIHW for NCHW and HWIO for NHWC
(I = C/groups). `MatMul(a, b)` with NumPy `matmul` semantics on rank ≥ 2
(batch dimensions broadcast).

After `passes.optimize`, 4-D activations are NHWC and every `Conv2d` and
pooling node has `layout="NHWC"`. Compute definitions must support NHWC;
NCHW support is optional.

## 4. Graph package

```python
frontend.onnx.load(model: str | Path | onnx.ModelProto,
                   input_shapes: Mapping[str, tuple[int, ...]] | None = None) -> Graph
ir.interpret(graph: Graph, inputs: Mapping[str, NDArray]) -> dict[str, NDArray]
passes.optimize(graph: Graph, *, layout: str = "NHWC", fold: bool = True) -> Graph
passes.fuse(graph: Graph, *, enabled: bool = True) -> list[FusedGroup]
passes.verify_groups(graph: Graph, groups: Sequence[FusedGroup]) -> None
passes.plan_memory(graph: Graph, groups: Sequence[FusedGroup], *,
                   alignment: int = 256) -> MemoryPlan
```

- **Import.** Dynamic dimensions are fixed by `input_shapes`. ONNX operators
  that only compute shapes (`Shape`, `ConstantOfShape`, `Range`, `Unsqueeze`
  of constants, ...) are evaluated at import time with the reference
  implementations. Operators with no Tessera equivalent are an error that
  names the operator and node.
- **Optimize** returns a new graph: canonical operators only (Gemm becomes
  MatMul + Add with the transpose folded into the weight; Flatten, Squeeze
  and Unsqueeze become Reshape; Identity and Dropout disappear), recognized
  patterns rewritten (decomposed LayerNorm and GELU), BatchNorm folded,
  constants folded, common subexpressions merged, dead nodes removed, and
  activations converted to NHWC with weights transformed at compile time
  and inverse transposes cancelled.
- **Fuse.** A group has one output and at most one REDUCTION/ANCHOR node.
  Injective and elementwise producers fuse into their consumer (prologue);
  elementwise/broadcast consumers fuse after an anchor (epilogue). A cheap
  producer with several consumers may be duplicated into each of them.
  `enabled=False` puts every node in its own group (for ablations).
  `pattern` is set from the anchor (Section 6). Groups made only of
  `Reshape` are views: they get no kernel, and their output shares the
  input's storage.
- **Memory planning.** Graph inputs and outputs are INPUT/OUTPUT refs,
  constants are packed into one blob (CONSTANT), and every other value
  read across groups gets an ARENA offset by greedy best-fit over its
  lifetime in group order. The plan reports the arena size, which the
  graph document compares with the sum of all intermediate tensors.

## 5. Loop IR (tir, te, compute)

The loop IR follows TVM's TensorIR: `PrimFunc` → loops (`For`) → blocks
(`BlockRealize` + `Block`) with iteration variables, read/write regions, an
optional reduction `init` and a body of `BufferStore`s
(`tessera/tir/stmt.py`). A function's body is a root block that allocates
its intermediate buffers.

```python
te.placeholder(shape, dtype, name) -> te.Tensor
te.compute(shape, fcompute, name) -> te.Tensor
te.reduce_axis(extent, name) -> IterVar;  te.sum / te.max / te.min
te.create_prim_func(params: Sequence[te.Tensor], name) -> PrimFunc
compute.lower_group(group: FusedGroup) -> LoweredGroup
tir.script(func) -> str                     # readable text form
tir.structural_hash(func) -> str            # names ignored; for tuning tasks
tir.interpret(func, args: Sequence[NDArray]) -> None   # writes outputs in place
tir.verify(func) -> None                    # well-formedness
tir.Schedule(func)                          # primitives in tessera/tir/schedule.py
tir.intrin.register(name, desc: PrimFunc, impl: ..., reference: ...)  # tensorize
```

- **lower_group** builds the group with `te`: prologue operators inline into
  the loads of the main block, the main block is the anchor or reduction,
  and the epilogue (if any) is one block. `LoweredGroup.dims` names the
  problem sizes a template needs.
- **Schedules preserve semantics.** Every primitive must leave the result
  unchanged; tests check this by interpreting functions before and after
  random sequences of primitives.
- **Tensorize** replaces a loop nest with a call to a hand-written
  microkernel. The intrinsic's description is a small PrimFunc; tensorize
  checks the nest computes the same thing before replacing it. The
  interpreter runs intrinsics through their Python reference.

## 6. Backends

Both backends implement `tessera.backend.base.Backend`: one template per
pattern in `tessera.plan.PATTERNS`, `build(plan, target)` and
`measure(func, target)`.

| Pattern | Groups |
|---|---|
| `conv2d` | Conv2d with groups == 1, plus epilogue |
| `depthwise_conv2d` | Conv2d with groups == channels |
| `matmul` | MatMul, plus prologue and epilogue |
| `pool` | MaxPool2d, AvgPool2d |
| `reduce` | ReduceSum, ReduceMean, ReduceMax |
| `softmax`, `layernorm` | Softmax, LayerNorm |
| `injective` | only elementwise, broadcast and injective nodes |

A template asks a `Config` for every decision (`tessera/tune/space.py`)
and gives good defaults, so untuned compilation is already fast.

**C backend.** One C11 translation unit per model: each kernel is a
function taking one `float*` (or other element pointer) per PrimFunc
parameter. Parallel loops are outlined and run on a persistent pthread pool
(performance cores by default). An executor function calls every kernel in
launch order with pointers into the inputs, outputs, arena and constant
blob, so a run is one native call. Targets: `c:generic` is portable C and
runs anywhere clang does (CI on x86-64 Linux); `c:neon` adds NEON
microkernels; `c:sme` adds SME microkernels (M4). Builds are cached by a
hash of the source and flags. `CompiledModel.sources()` returns the C file.

**Metal backend.** Kernels are MSL, compiled at run time with
`newLibraryWithSource` (no Xcode needed). A small Objective-C shim with a C
ABI, compiled by clang on first use, owns the device, pipelines and
buffers. Thread-bound loops map to the grid; `shared` buffers become
threadgroup memory with barriers inserted. A run encodes every launch into
one command buffer. Inputs and outputs live in shared-storage buffers
(unified memory), so no copies to a device are needed.

## 7. Tuning

A tuning task is a lowered group's structure (its structural hash) plus the
template and the target. The tuner builds the task's `ConfigSpace` from the
template, proposes configurations, builds candidates in parallel processes,
measures them one at a time (`backend.measure`, with a timeout), and records
every result in a JSON-lines log:

```json
{"target": "c-sme-t4-Apple_M4", "task": "matmul:3f2a...", "config": {"tile_m": [8, 16]},
 "seconds": 0.000412, "error": null, "time": "2026-11-12T10:00:00Z"}
```

`compile()` uses the best logged configuration of each task and the
template defaults for tasks with none. Logs for the benchmark models are
committed under `results/tuning/`, so a fresh clone compiles tuned models
without tuning. Search strategies: random, and model-guided evolutionary
search with a cost model learned during the search from features of the
scheduled loop nests. The tuning document compares them (best time found
against measured trials).

## 8. Driver and command line

```python
tessera.compile(model, *, target="host", input_shapes=None,
                tuning_log=None, fuse=True, optimize=True) -> CompiledModel
```

```
tessera compile MODEL [--target T] [--shape NAME=1,3,224,224] [-o DIR]
tessera run MODEL [--target T] [--check]      # random inputs, compared with the interpreter
tessera inspect MODEL --stage graph|optimized|fused|tir|code [--group NAME]
tessera profile MODEL [--target T]            # per-kernel times
tessera tune MODEL [--target T] [--trials N] [--strategy model|random] [--log FILE]
tessera bench models|ops|tuning|all [--target T]   # writes results/*.md and *.json
```

## 9. Evaluation

Measured on an Apple M4 (4 performance + 6 efficiency cores, 10-core GPU,
16 GB), fp32, batch 1 unless noted:

- **Models:** ResNet-50 and MobileNetV2 (torchvision, 224×224), BERT-base
  (sequence 128) and GPT-2 small (prefill of 128 tokens), with their real
  pretrained weights, exported to ONNX by `scripts/export_models.py`.
- **Baselines:** PyTorch eager on the CPU and on MPS, ONNX Runtime on the
  CPU. Latency is the median of at least 50 runs after warmup, with p10/p90.
- **Correctness:** maximum absolute and relative error of every output
  against ONNX Runtime, and top-1 agreement for the image models.
- **Kernels:** GEMM throughput (NEON, SME, Metal) against Accelerate
  (through NumPy) and MPS; ResNet-50 convolution layers.
- **Ablations:** fusion off/on (kernel count, memory traffic, latency),
  memory planning (arena size), tuning (default vs tuned, random vs model).

Benchmark numbers are only recorded from a quiet machine. Every number in
the README comes from a file in `results/` that a command regenerates.

## 10. Testing

`make check` runs ruff, `mypy --strict` and pytest; tests never touch the
network. Operators are tested against ONNX Runtime; passes by interpreting
graphs before and after (including Hypothesis-generated graphs); schedule
primitives by interpreting functions before and after random schedules;
each backend by compiling every pattern on a range of shapes and comparing
with the reference; the driver end to end on small models built in the
tests. Tests that need a Metal device or SME skip themselves where those
are missing. Real models are marked `models` and run with `-m models`.
Floating-point comparisons use a tolerance relative to a float64 reference,
because fused and reordered reductions round differently.

## 11. Conventions

- Python 3.12+, `mypy --strict`, ruff, 100 columns. Public functions have
  docstrings that say what they do.
- Generated C and MSL is readable: named after the graph nodes, indented,
  and stable for the same input (so builds cache and diffs are meaningful).
- No network access outside `scripts/export_models.py` and the `bench`
  dependency group.
- Every claim in a document is backed by a test or a file in `results/`.
