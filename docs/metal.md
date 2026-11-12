# The Metal backend

`tessera.backend.metal` runs Tessera's kernels on the Apple GPU. It turns
each scheduled PrimFunc into a Metal Shading Language (MSL) kernel, compiles
all of a model's kernels into one library at run time, and runs a model by
encoding every kernel launch into a single command buffer. A small
Objective-C runtime with a C ABI, compiled by clang on first use, owns the
device, the compiled pipelines and the buffers; Python drives it through
ctypes. Schedule templates (`templates.py`) decide how each pattern of
`tessera.plan.PATTERNS` maps onto threadgroups and threads.

Nothing here needs Xcode: the Command Line Tools provide clang, and
`MTLDevice newLibraryWithSource:` compiles MSL at run time. Everything is
measured on an Apple M4 with a 10-core GPU. Tests that need a GPU skip
themselves when there is none (CI runners have none).

| Module | What it does |
|---|---|
| `codegen.py` | PrimFunc → MSL kernel: grid mapping, buffers, barriers, intrinsics |
| `lower.py` | IR rewrites before printing: buffer compaction, cross-thread reductions |
| `shim.m`, `runtime.py` | the Objective-C runtime and its ctypes wrapper |
| `executor.py` | `MetalModel`: a whole `ExecutionPlan` in one command buffer |
| `measure.py` | timing one kernel for the tuner |
| `templates.py` | one schedule template per pattern |
| `bench.py` | the kernel benchmarks in `results/metal/` |

## Generated code

This is the kernel the softmax template generates for a 12 x 128 softmax
over the last axis (`tests/metal/test_metal_docs.py` checks that it is
exactly what the code generator prints today):

```metal
// softmax: 12x1x1 threadgroups of 128 threads
[[max_total_threads_per_threadgroup(128)]]
kernel void softmax(
    device const float* x [[buffer(0)]],
    device float* softmax [[buffer(1)]],
    uint3 threadgroup_pos [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]) {
  threadgroup float softmax_sum_shared[1];
  threadgroup float softmax_max_shared[1];
  threadgroup float softmax_max_shared_red[4];
  threadgroup float softmax_sum_shared_red[4];
  float softmax_max_shared_partial;
  float softmax_sum_shared_partial;
  const int group_x = (int)threadgroup_pos.x;
  const int thread_x = (int)thread_index;
  softmax_max_shared_partial = -INFINITY;
  // softmax_max
  softmax_max_shared_partial = max(softmax_max_shared_partial, x[group_x * 128 + thread_x]);
  // max across the 128 threads of thread.x
  softmax_max_shared_partial = simd_max(softmax_max_shared_partial);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_x % 32 == 0) {
    softmax_max_shared_red[thread_x / 32] = softmax_max_shared_partial;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  softmax_max_shared_partial = softmax_max_shared_red[0];
  for (uint group = 1; group < 4; ++group) {
    softmax_max_shared_partial = max(softmax_max_shared_partial, softmax_max_shared_red[group]);
  }
  // softmax_max_write
  if (thread_x == 0) {
    softmax_max_shared[0] = softmax_max_shared_partial;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  softmax_sum_shared_partial = 0.0f;
  // softmax_sum
  softmax_sum_shared_partial += exp(x[group_x * 128 + thread_x] - softmax_max_shared[0]);
  // add across the 128 threads of thread.x
  softmax_sum_shared_partial = simd_sum(softmax_sum_shared_partial);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_x % 32 == 0) {
    softmax_sum_shared_red[thread_x / 32] = softmax_sum_shared_partial;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  softmax_sum_shared_partial = softmax_sum_shared_red[0];
  for (uint group = 1; group < 4; ++group) {
    softmax_sum_shared_partial = softmax_sum_shared_partial + softmax_sum_shared_red[group];
  }
  // softmax_sum_write
  if (thread_x == 0) {
    softmax_sum_shared[0] = softmax_sum_shared_partial;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // softmax
  softmax[group_x * 128 + thread_x] = exp(x[group_x * 128 + thread_x] - softmax_max_shared[0]) / softmax_sum_shared[0];
}
```

`compute.lower_group` produced three blocks: the row maximum, the sum of
exponentials and the normalized output. The template gave each row a
threadgroup, bound the output's columns and both reductions' loops to the
128 threads, and moved both statistics into threadgroup memory under the row
loop. Everything after that is the code generator: the two reductions bound
to threads became SIMD-group reductions combined through `*_red`, the
barriers come from the barrier pass, and the comments name the blocks of the
loop IR.

## From loop IR to MSL

Every thread of a launch runs the whole kernel body, so a loop bound to a
grid axis is not a loop on the GPU but the thread's position on that axis.

**Grid axes.** A loop bound to `threadgroup.{x,y,z}` reads
`threadgroup_position_in_grid`; its extent is the number of threadgroups on
that axis. Loops bound to `thread.{x,y,z}` give the threadgroup's shape. Each
axis gets one variable (`group_x`, `thread_y`, ...) that every loop bound to
it reads, so the many thread loops a schedule creates (one per cooperative
copy, per init, per store) all print as the same name. Two loops bound to one
axis may have different extents: the launch uses the largest and the smaller
loop runs under `if (thread_x < extent)`.

Threadgroups are launched one-dimensional, and the code computes each axis
position from `thread_index_in_threadgroup`, x fastest (`thread_index % 32`,
`thread_index / 32`). `generate(func, flat=False)` launches the threadgroup in
the thread loops' shape instead; `results/metal/shape.md` compares the two on
SIMD-group matrix multiplies, where they perform within a few percent of each
other.

**Buffers.** A function's parameters become `device` pointers, `const`
when the kernel never writes them, bound at buffer indices 0, 1, .... Buffers
the function allocates become threadgroup arrays (`shared`), thread-private
arrays (`local`) or scratch device buffers bound after the parameters
(`global`). Threadgroup arrays must be declared at kernel scope in MSL, so
all allocations are hoisted there. A global intermediate is only accepted in
a kernel with one threadgroup, because threads of different threadgroups
cannot wait for each other inside a kernel.

**Compaction.** Schedules create cache buffers with the shape of the whole
tensor and then move the blocks that use them under tile loops. On the GPU a
local buffer exists once per thread and a shared buffer once per
threadgroup, so `lower.compact_buffers` shrinks each one to the region one
thread (or one threadgroup) touches: it finds the innermost statement around
all uses and, holding fixed the loops around it plus every thread-bound loop
(local) or the threadgroup-bound loops (shared), takes the bounds of each
access over the remaining loops. A matmul's `C_local` of 64 x 64 becomes the
2 x 2 accumulators of one thread, and its `A_shared` the 16 x 8 tile of one
step (`tests/metal/test_metal_lower.py`). Indices are split into additive
terms, so an affine fixed part (`by * 16`) and a gathered part
(`flat // 8`) are handled separately. Loops bound to the same axis count as
the same position. The tir package's own `compact_buffers` places buffers by
loop nest only; the Metal backend needs the scope-dependent version.

**Barriers.** A pass over each statement sequence inserts
`threadgroup_barrier(mem_flags::mem_threadgroup)` before a statement that
reads a shared buffer an earlier statement wrote, or writes one an earlier
statement read or wrote, since the last barrier. Each statement is
summarized by the accesses before its first barrier and after its last, so
barriers inside nested loops count. A loop whose body ends with accesses
that conflict with its beginning (the tile copy of the next iteration
overwriting the tile the current one reads) gets a barrier at the end of its
body. A barrier inside a branch that depends on the thread would deadlock or
be undefined, so the code generator refuses one there.

**Reductions across threads.** When a schedule binds a loop that drives a
block's reduction variable to `thread.x`, `lower.lower_cross_thread_reductions`
rewrites the block: each thread accumulates a partial over its share (threads
beyond the loop's extent contribute the identity), the partials of the
threads that share `thread.y` and `thread.z` are combined with
`simd_sum`/`simd_max`/`simd_min` (or `simd_shuffle_xor` steps when fewer than
32 threads share a row), and rows wider than a SIMD group combine their SIMD
groups' results through a small threadgroup array. One thread writes a shared
or global result; every thread writes a local one, since each thread holds
its own copy. Sums, maxima and minima are supported, over widths that are a
power of two up to 32 or a multiple of 32 (`tests/metal/test_metal_reduce.py`
runs widths 1 to 1024, odd lengths and all three operators on the GPU).

**SIMD-group matrices.** `metal.simdgroup_fill`, `_load`, `_store` and `_mma`
work on 8 x 8 tiles of local buffers that only these calls use; such buffers
are declared as arrays of `simdgroup_float8x8`, compacted like any local
buffer, and a kernel that uses them runs the 32 lanes of a SIMD group on
`thread.x` while `thread.y` and `thread.z` index SIMD groups. Each call also
carries the row stride of its tiles, which the interpreter's references need;
the generated code addresses tiles by index instead.

**Arithmetic.** Index arithmetic uses 32-bit `int` when every buffer size,
loop extent and constant fits, which makes division by constants cheap;
otherwise `long`. `floordiv` and `floormod` print as `/` and `%` when both
operands are provably non-negative and as small helper functions otherwise.
Kernels are compiled with `MTLMathModeRelaxed`: fast floating-point
optimizations that keep infinities and NaNs, since reductions start from
`-INFINITY`. Two math functions needed care: fast-math `tanh` returns NaN for
inputs above about 44.5, so the generator clamps its argument to ±9 (where
`tanh` is ±1 in float32), and MSL has no `erf`, so a helper implements
Abramowitz and Stegun 7.1.26 (absolute error below 1.5e-7). `pow` with a
negative base and an integral exponent follows NumPy. Generated code is
deterministic: names come from the IR (sanitized, made unique, never shadowing
a Metal function), and the same function always prints the same text.

## The runtime

`shim.m` is about 400 lines of Objective-C compiled with
`clang -fobjc-arc -O2 -dynamiclib -framework Metal -framework Foundation` into
`~/.cache/tessera/metal/` (or `$TESSERA_CACHE/metal/`), named by a hash of its
source and flags, so it is built once per version; concurrent processes
compile into private files and rename them into place. Every object it hands
to Python is a retained Objective-C object behind a `void*`, released by a
`weakref.finalize` on the Python wrapper.

- **Libraries** are compiled from source with the compiler's messages
  returned as the error text (`MetalError`).
- **Buffers** use shared storage, the unified memory both the CPU and the GPU
  address, so `Buffer.view` gives NumPy arrays over a buffer's bytes with no
  copies. Buffers are created untracked: Metal's automatic hazard tracking
  would only add cost, because programs order their own launches.
- **Programs** are lists of launches: a pipeline, a grid, a threadgroup
  shape, (buffer, offset) pairs and whether the launch waits for the earlier
  ones. `tm_program_run` encodes all launches into one command buffer with one
  concurrent compute encoder and a `memoryBarrierWithScope:` before each
  launch marked as dependent. The command buffer does not retain resources;
  the program does.
- **Profiling** (`tm_program_profile`) gives each launch its own compute pass
  in one command buffer, orders the passes with a fence, and has the GPU
  sample timestamps at the start and end of each pass
  (`MTLCounterSamplingPointAtStageBoundary`). GPU ticks are converted to
  seconds from two CPU/GPU timestamp pairs taken around the run.
- **Failures and timeouts.** A failed command buffer's error text, with the
  label of the faulting pass when Metal reports it, becomes a `MetalError`. A
  run waits for completion on a semaphore with a timeout; on timeout the
  program stays alive until the GPU finishes with it.

## Running a model

`MetalModel` (`executor.py`, returned by `backend().build(plan, target)`)
generates every kernel of an `ExecutionPlan` into one MSL source
(`sources()["kernels.metal"]`), compiles one library, and allocates one buffer
each for the arena, the constant blob, every model input and every model
output. Each launch binds its parameters to (buffer, offset) pairs taken from
the plan's `BufferRef`s, after checking that every region fits.

A launch gets a barrier only if it reads bytes written, or writes bytes read
or written, by a launch since the last barrier (`plan_barriers`, checked by a
Hypothesis test that every pair of conflicting launches has a barrier
between them). Independent kernels, such as the two branches of a residual
block, may then overlap on the GPU. `run` copies the inputs into their
buffers, runs the program and copies the outputs; `input_views` and
`output_views` expose the buffers themselves for callers that want no copies.
`benchmark` returns wall-clock seconds per run (encoding to completion) after
warming the GPU up, `gpu_benchmark` the GPU time of each command buffer, and
`profile` the median GPU time of every launch.

## Timing kernels for the tuner

`backend().measure(func, target)` generates and compiles `func`, fills its
buffers with values every operator accepts (floats in [0.5, 1.5), integer
zeros as indices), and returns the median GPU time per launch over `repeat`
command buffers. Each command buffer holds a batch of launches separated by
barriers, sized so that all batches together take about `min_seconds`; this
hides the fixed cost of a command buffer behind many launches. Code
generation and compilation errors become `MeasureError`. `prebuild` does the
generation and compilation without running, so the tuner's parallel build
processes report broken candidates.

Before timing, `measure` keeps the GPU busy for 50 ms in one command buffer.
An idle M4 GPU runs at a low clock and speeds up only under sustained load:
in `results/metal/warmup.md`, a compute-bound kernel run in batches with the
CPU waiting for each batch took 1592 µs per launch in the first batch after
3 s of idle time, 612 to 616 µs in the next two, and 388 µs once about 37 ms
of work had gone through; after one continuous command buffer of 24 ms it took
386 to 413 µs from the first batch. Without the warmup, the tuner would compare
candidates measured at different clocks.

## Measurements

All files in `results/metal/` record the date, machine, commit and load
average of their run, and are marked provisional when the machine was busy.

**The cost of a launch** (`results/metal/launch.md`, a one-threadgroup kernel
launched up to 1024 times in one command buffer): a launch that waits for the
previous one costs about 3.2 µs of GPU time with the concurrent encoder and
2.8 µs with Metal's serial encoder; independent launches cost 0.2 µs. A
command buffer with one launch takes about 180 µs from encoding until the CPU
sees it complete, which is why a model runs as one command buffer.

## Limitations

- Kernels compute in float32 (float16 buffers are read and converted).
- A function may bind at most 31 buffers (Metal's limit for direct
  arguments); groups with more inputs fail to generate.
- Global intermediates are refused in kernels with more than one
  threadgroup.
- Cross-thread reductions run over `thread.x` only and support sums, maxima
  and minima.
- The code generator assumes the 32-lane SIMD groups of Apple GPUs.
