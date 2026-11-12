# The C backend

`tessera.backend.c` turns the scheduled loop nests of a model into one C11
file, compiles it with clang into a shared library and runs the whole model
with one native call. It has three targets: `c:generic` (portable C that
any clang target runs, including Linux x86-64 in CI), `c:neon` (tuned for
the host Arm core, with a hand-written NEON GEMM and convolution) and
`c:sme` (the M4, whose matrix products run on its SME unit). The package
provides, as `tessera.backend.base.Backend` asks:

- `templates(target)`: one schedule template per pattern, specialized for
  the target's ISA and thread count;
- `build(plan, target)`: the compiled model (`CModel`);
- `measure(func, target, constants=...)` and `prebuild(func, target,
  constants=...)`: timing of one kernel in a separate process, for the
  tuner; `constants` names the parameters that hold model weights.

`run_kernel(func, arrays, target)` compiles one PrimFunc and runs it on
NumPy arrays in place, the native counterpart of `tir.interpret`; the tests
use it to check every scheduled kernel.

## Generated code

This is the kernel the `matmul` template generates for `relu(x @ w + b)`
with `x` 64 x 256 and `w` 256 x 128, for `c:generic` with four threads
(printed by `tessera inspect MODEL --stage code`, or `codegen.generate_kernel`):

```c
typedef struct {
  float* x;
  float* w;
  float* b;
  float* relu;
} matmul_parallel0_args;

static void matmul_parallel0_body(float* restrict x, float* restrict w, float* restrict b, float* restrict relu, int64_t begin, int64_t end) {
  for (int64_t n_0 = begin; n_0 < end; ++n_0) {
    {
      float w_local[4096] __attribute__((aligned(64)));
      for (int64_t ax0_1 = 0; ax0_1 < 256; ++ax0_1) {
        for (int64_t ax1_1 = 0; ax1_1 < 16; ++ax1_1) {
          {  // w_local
            const int64_t v0_1 = ax0_1;
            const int64_t v1_1 = n_0 * 16 + ax1_1;
            w_local[ax0_1 * 16 + ax1_1] = w[v0_1 * 128 + v1_1];
          }
        }
      }
      for (int64_t m_0 = 0; m_0 < 16; ++m_0) {
        {
          float matmul_local[64] __attribute__((aligned(64)));
          for (int64_t m_1_init = 0; m_1_init < 4; ++m_1_init) {
            for (int64_t n_1_init = 0; n_1_init < 16; ++n_1_init) {
              {  // matmul_init
                const int64_t v_m_1 = m_0 * 4 + m_1_init;
                const int64_t v_n_1 = n_0 * 16 + n_1_init;
                matmul_local[m_1_init * 16 + n_1_init] = 0.0f;
              }
            }
          }
          for (int64_t k = 0; k < 256; ++k) {
            #pragma clang loop unroll(full)
            for (int64_t m_1 = 0; m_1 < 4; ++m_1) {
              #pragma clang loop vectorize(enable) interleave(enable)
              for (int64_t n_1 = 0; n_1 < 16; ++n_1) {
                {  // matmul
                  const int64_t v_m = m_0 * 4 + m_1;
                  const int64_t v_n = n_0 * 16 + n_1;
                  const int64_t v_k = k;
                  matmul_local[m_1 * 16 + n_1] += x[v_m * 256 + v_k] * w_local[k * 16 + n_1];
                }
              }
            }
          }
          for (int64_t ax0 = 0; ax0 < 4; ++ax0) {
            #pragma clang loop vectorize(enable) interleave(enable)
            for (int64_t ax1 = 0; ax1 < 16; ++ax1) {
              {  // matmul_local
                const int64_t v0 = m_0 * 4 + ax0;
                const int64_t v1 = n_0 * 16 + ax1;
                relu[v0 * 128 + v1] = fmaxf(matmul_local[ax0 * 16 + ax1] + b[v1], 0.0f);
              }
            }
          }
        }
      }
    }
  }
}

static void matmul_parallel0(void* args, int64_t begin, int64_t end) {
  const matmul_parallel0_args* a = (const matmul_parallel0_args*)args;
  matmul_parallel0_body(a->x, a->w, a->b, a->relu, begin, end);
}

static void matmul(float* restrict x, float* restrict w, float* restrict b, float* restrict relu) {
  {
    matmul_parallel0_args args = {x, w, b, relu};
    tsr_parallel_for(0, 8, matmul_parallel0, &args);
  }
}
```

The printer follows the loop IR without restructuring it:

- **Kernels** are `static void <group>(T* restrict <param>, ...)`, one
  pointer per PrimFunc parameter, named after the graph values. `restrict`
  is sound because the memory planner never gives two live values the same
  bytes, and it lets clang vectorize without runtime alias checks.
- **Blocks** open a scope that binds their iteration variables (`v_m`,
  `v_n`, `v_k`), tests their predicate (imperfect splits) and runs their
  reduction `init` when every reduce variable is at its first value. After
  `decompose_reduction` the init is a block of its own (`matmul_init`).
- **Local buffers** are stack arrays. `cache_write` and `cache_read` create
  them with the shape of the whole tensor; `alloc.plan_allocations` places
  each one at the innermost statement that contains all its uses and
  shrinks it to the region those uses touch there. Here the 64 x 128
  accumulator becomes `matmul_local[64]`, one 4 x 16 tile per iteration of
  the parallel loop, so each thread has its own. A buffer written by a
  reduction is placed above the reduction's outermost loop, since its value
  lives across that loop. Index terms built only from loops outside the
  allocation (such as `t >> 3` when the tile loops are fused into `t`) are
  constant while the buffer lives and drop out of the shrunk index. Buffers
  above 64 KiB live in storage each thread allocates once and reuses (the
  calling thread works on every parallel loop, and Python
  threads other than the main one have 512 KiB stacks).
- **Global intermediates** (a main block's result that an epilogue could not
  be fused into) live in storage each calling thread allocates once and
  keeps; allocating megabytes per call costs page faults every run.
- **Parallel loops** are outlined: the body becomes `<kernel>_parallelN_body`
  with `restrict` parameters for everything it uses from outside, a
  trampoline unpacks an argument struct, and the kernel hands
  `[begin, end)` to `tsr_parallel_for`, or to `tsr_parallel_for_n` with a
  thread count when the loop carries the `parallel.threads` annotation (the
  SME schedules use it). A parallel loop inside another runs serially.
- **Vectorized and unrolled loops** carry `#pragma clang loop
  vectorize(enable) interleave(enable)` and `unroll(full)`. The templates
  only vectorize innermost loops with unit-stride accesses and no
  predicates, which clang vectorizes; `test_vectorized_loop_compiles_to_simd_instructions`
  checks the assembly (NEON `fmla.4s` on Arm, packed SSE on x86).
- **Indices** are flattened row-major with small constant folding.
  `floordiv`/`floormod` by a power of two become `>>` and `&` (correct for
  negative values), by other constants `/` and `%` when the numerator is
  known non-negative (loop and block variables with non-negative bindings),
  and a helper with Python's rounding otherwise. A loop of one iteration is
  replaced by its value, and so is a block variable bound to a constant.
- **Math intrinsics** call the runtime's own `exp`, `log`, `tanh`, `erf`
  and `sigmoid` (below), because a call into libm keeps clang from
  vectorizing the loop around it.
- **Microkernels**: `Evaluate(Call("c.<name>", args))` calls a registered C
  function, after checking that the target's ISA has it.

The output depends only on the function, so equal functions print equal
text, hit the same build cache entry and diff meaningfully.

## Runtime

`include/tessera_runtime.h` is included once into every generated file;
everything in it is `static`, so each loaded model has its own pool.

**Thread pool.** `TSR_NUM_THREADS - 1` persistent workers (the thread
count is part of the source, so single- and multi-threaded builds of a model
are different libraries). `tsr_parallel_for(begin, end, task, args)` runs
the range on every thread, the caller included: threads claim iterations
from a shared counter an eighth of an even share at a time (one iteration
when there are few, as for the tiles the templates parallelize), so a
thread that runs slower than the others takes fewer. A loop is published by
writing the job and incrementing a generation counter; workers poll the
counter for 0.5 ms (a model's kernels follow each other within
microseconds, so workers stay awake through a run) and then sleep on a
condition variable. On SME targets they sleep at once (`TSR_SPIN_SECONDS`
0, see "Threads on SME targets" below). The publisher takes the mutex only
when some worker sleeps; the sleeper registers itself before rechecking the
counter under the mutex, so a wakeup cannot be lost. Every worker
acknowledges every job, so the job fields are never rewritten while a
worker may still read them. `tsr_parallel_for_n` limits a loop to fewer
threads (the others acknowledge and stay idle). On macOS the workers ask for
`QOS_CLASS_USER_INTERACTIVE`, which keeps them on the performance cores (the
default thread count is the number of performance cores), and so does every
exported entry point for its calling thread, once: the caller runs the
serial parts and works on every loop, and a kernel that never starts the
pool (single-threaded SME) otherwise ran at the caller's default class,
which macOS may move to an efficiency core. A worker that wakes up on an
efficiency core for a loop leaves the loop to the others and sleeps until
the next one (on Apple silicon the efficiency cores are the first
`hw.perflevel1.logicalcpu` logical CPUs; `pthread_cpu_number_np` says which
one a thread is on). A second thread starting a loop while the pool is busy
runs it serially, and after `fork()` the child runs every loop serially.
`test_thread_pool` checks 2000 loops of varied sizes for exactly-once
execution, wakeups after sleeping, nesting, concurrent callers and fork.

**Threads on SME targets.** The performance cores share one SME unit, and
the efficiency cluster has its own. When several threads keep SME busy,
macOS moves some of them to the efficiency cluster (its scheduler spreads
users of a cluster's shared unit across clusters), whatever their QoS
class; NEON kernels alone never moved a thread. There an SME chunk takes
about three times as long: a 128 x 2304 x 768 product split in three
chunks took 165 µs per chunk on a performance core and 465 µs on an
efficiency core, and a loop split into one chunk per thread waits for the
slowest. Instrumented runs of ResNet-50 (untuned, c:sme) with polling
workers fell into three groups by the share of chunks that ran on
efficiency cores: none, 12–14 ms; a quarter (one thread), 17.5–18.5 ms;
half, 24.5–25 ms. That was the spread of its untuned latency (18.0 to 24.7
ms from p10 to p90). Three changes answer it:

- On SME targets idle workers sleep at once instead of polling: a polling
  worker never blocks, so it stayed where it was moved for the rest of the
  run, NEON kernels included, while every wakeup lets the scheduler place a
  sleeping worker again. Polling stays on NEON and generic targets, where a
  wakeup (3–7 µs) would cost more than the loops it parallelizes
  (MobileNetV2 on c:neon: 3.0 ms polling, 4.2 ms sleeping).
  `runtime_defines(target)` in `build.py` writes these settings into every
  generated source.
- A worker woken on an efficiency core skips the loop, so it is never the
  chunk everyone waits for.
- Iterations are claimed, not split in advance, so the threads that work
  share the loop between them.

Sleeping alone left about half the runs of some processes with a thread on
an efficiency core (medians 13.6 and 18.5 ms in two of three processes);
with all three, three processes ran ResNet-50 in 11.8–11.9 ms (p90 at most
12.2 ms) with no chunk on an efficiency core. The scheduler can still keep
one worker on the efficiency cluster for a whole process, waking it there
for every loop; the model then runs on three threads, 15.5 ms instead of
11.8 ms (and instead of 17.5–25 ms before). NEON targets are unchanged
(MobileNetV2, ResNet-50 and BERT on c:neon within 1 percent).

**Memory and time.** `tsr_alloc` returns 128-byte-aligned memory (the cache
line of Apple's performance cores); `tsr_now` reads `mach_absolute_time`
(or `CLOCK_MONOTONIC`).

**Math.** Branch-free single-precision functions that clang vectorizes: all
cases are computed and the result picked with selects.

| function | method | error (tests/c/test_c_math.py) |
|---|---|---|
| `exp` | Cody–Waite reduction, Cephes degree-6 polynomial, 2^n by adding to the exponent field | ≤ 1.5 ulp |
| `log` | mantissa in [√½, √2), Cephes polynomial | ≤ 1 ulp |
| `tanh` | x·P(x²)/Q(x²) on [−7.9, 7.9] | ≤ 5 ulp, < 4e-7 absolute |
| `erf` | x·P(x²)/Q(x²) on [−4, 4] | ≤ 6 ulp, < 4e-7 absolute |
| `sigmoid` | 1 / (1 + exp(−x)) | ≤ 3 ulp |

The rational forms of `tanh` and `erf` are the ones Eigen uses; they trade
a few ulp near ±1 (absolute error stays below 4e-7, which is what GELU
sees) for a dozen FMAs and one division instead of an exponential. 2^n is
applied by an
integer add rather than by multiplying by 2^n1·2^n2, because
`-fassociative-math` may regroup that product into an overflowing 2^128.

## Build and execution

**One translation unit per model.** `executor.model_sources(plan, target)`
writes `model.c`: `#define TSR_NUM_THREADS`, the runtime and microkernel
headers, every kernel in launch order of first use, and three exported
functions:

```c
void tessera_run(void* const* inputs, void* const* outputs,
                 uint8_t* arena, uint8_t* constants);
void tessera_profile(..., double* seconds);           /* time of each launch */
void tessera_benchmark(..., int32_t warmup, int32_t repeat, double* seconds);
```

`tessera_run` calls the kernels in launch order; each argument is a pointer
derived from a `BufferRef` (`(float*)inputs[0]`, `(float*)(arena + 512)`,
`(float*)(constants + 256)`), so a run is one native call with no Python in
between. `CModel` (the `CompiledModel`) owns the arena and a 128-byte-aligned
copy of the constant blob, loads the library with ctypes, makes inputs
contiguous, returns new output arrays (copying an output that the plan keeps
elsewhere), and serializes runs with a lock. `benchmark` times runs inside
C; `profile` returns the median time of every launch.

**Build cache.** `build.build(sources, main, target)` hashes the compiler
version, the flags and every source file; the binary lives in
`$TESSERA_CACHE/c/<hash>/` (default `~/.cache/tessera/c`) next to the
sources and the command that built it. A build happens in a temporary
directory that is renamed into place, so concurrent builds of the same key
(the tuner builds in parallel processes) never see a partial file. Flags:

| | flags |
|---|---|
| all | `-std=c11 -O3 -ffp-contract=fast -fno-math-errno -fno-trapping-math -fno-signed-zeros -fassociative-math -fvisibility=hidden` |
| `generic` | nothing else (clang's default CPU) |
| `neon` | `-mcpu=native` |
| `sme` | `-mcpu=apple-m4` |

Reassociation lets clang vectorize reductions; `-ffinite-math-only` is left
out because max-reductions start from `-INFINITY`. The M4 runs SVE
instructions only in streaming mode; `apple-m4` enables SME and SME2 but not
SVE, so ordinary code stays NEON and only functions marked
`__arm_locally_streaming` use streaming SVE.

**Measuring a kernel.** `measure(func, target)` compiles the kernel with a
`main` that fills every float parameter with random values (integer and
boolean ones with zeros, which are always valid indices), calibrates how
many calls make one sample, and prints the per-call time of each sample;
the median is returned. The calls are timed as a model run makes them. The
parameters listed in `constants` (indices of model weights) get as many
copies as it takes to span 64 MiB (at most 64), more than the caches; every
copy is prepared by one untimed call, as a model's first run packs its
weights, and the timed calls rotate through the copies, so each reads its
weights from memory as a model run does. The other parameters stay in one
warm copy, like the activations the previous kernel just wrote. On GPT-2's
QKV projection (128 x 768 by 768 x 2304, 12 layers) the template's default
took 474 µs in `measure` and 498 µs in the model, and 32 x 64 tiles 620 µs
and 692 µs. Before the SME threads slept, `measure` gave 1063 and 638 µs
for the same two: it ranked them the other way round. A warm weight alone
made `measure` up to a fifth faster than the model
(`test_measure_ranks_weight_configs_as_the_model_does`). The harness runs as a subprocess, so a candidate
that crashes or hangs becomes a `MeasureError` (as do build errors and
timeouts) and cannot take down the tuner. `prebuild` compiles the same
harness into the cache from the tuner's build processes, so the later
`measure` only runs it.

## Templates

Every template asks its `Config` for each decision and asks every knob on
every call, whatever the other knobs' values: the space is recorded from one
run with the defaults, and every point must answer the same questions.
Defaults are chosen so that untuned compilation is fast.

Every template ends by fusing its outer loops and running them on the
thread pool when the knob `parallel` says so (by default when the kernel
does at least 2^15 multiply-adds or elements). Tile options are divisors of
the extents, so tiles never need predicates, which would keep clang from
vectorizing.

| pattern | schedule | knobs (default) |
|---|---|---|
| `matmul` | 4 x 16 register tiles of C accumulated in a local buffer over the whole reduction (rows unrolled, columns vectorized); column strips outside row tiles, each strip of B packed once into a contiguous local copy when it fits the stack (64 KiB); epilogue on each tile | `tile_m` (4), `tile_n` (≤ 16, a multiple of 4 when one divides N), `order` (`nm`; `mn` puts row tiles outside), `pack_b` (on) |
| `matmul`, NEON/SME | tiles of C (edge tiles allowed) computed by the GEMM microkernel, the reduction's init before each call and the epilogue after it | `kernel` (`micro`), `micro_m` (≤ 256 rows), `micro_n` (a whole number of tiles per thread, ≤ 1024 columns; on SME sized for one thread), `micro_threads` (all; on SME all only when there are no more tiles than threads, else 1) |
| `conv2d` | as matmul with output columns as rows: `tile_ow` columns x `tile_c` channels in registers over KH·KW·C; channel strips outside pixel tiles, each strip's weights packed when they fit the stack | `tile_ow` (4), `tile_c` (16), `order` (`co`), `pack_w` (on) |
| `conv2d`, NEON/SME | strips of output channels computed by the implicit-GEMM microkernel over the whole output plane, bias (the reduction's init) before the call, epilogue after | `kernel` (`micro`), `micro_c` (a whole number of strips per thread, ≤ 256 channels), `micro_threads` (all) |
| `depthwise_conv2d` | `tile_ow` columns x `tile_c` channels accumulated over the window | `tile_ow` (4), `tile_c` (16) |
| `pool` | channels vectorized, the window reduced in a local buffer, the average's scaling on the stored tile | `tile_ow` (1), `tile_c` (16) |
| `reduce` | over the last axes: one accumulator per output element, the contiguous loop vectorized by clang; over middle axes (a global average pool): a `tile` of the innermost output axis accumulated while the reduction streams over contiguous rows | `tile` (the whole axis, or one tile per thread when the outer axes are too few) |
| `softmax`, `layernorm` | every statistics stage computed per row inside the final stage's row loop, reductions decomposed so clang vectorizes them, the final stage's row loop vectorized | `vector` (16) |
| `injective` | stages inlined into the output's, its innermost loop vectorized | `vector` (16) |

A matmul operand that prologue operators compute is lowered as a block of
its own (`CBackend.lowering_options` asks `compute.lower_group` for
`operand_blocks`). The matmul template inlines it back when it copies a
strided matrix of its source (a slice, a transpose of axes: the loads stay
affine and the microkernels read the source in place) and otherwise computes
it once, element by element, in a nest of its own before the product (no
knobs, vectorized, parallel). The attention output projection of BERT and
GPT-2 reads its operand as `x[k // 64, m, k % 64]` (heads transposed and
merged); inlined, that index arithmetic ran once per multiply-add in the
generated loops, since the microkernels need a plain matrix. Staged, the
twelve projections of GPT-2 take 1835 µs instead of 5616 µs (1862 µs for
the unfused transpose, reshape, product and additions), and fusion stopped
costing time on c:sme:

| c:sme, untuned | fused before | fused | fusion off |
|---|---:|---:|---:|
| GPT-2 | 37.4 ms | 32.3 ms | 32.2 ms |
| BERT | 26.8 ms | 22.0 ms | 22.5 ms |
| ResNet-50 | | 12.0 ms | 11.9 ms |
| MobileNetV2 | | 3.34 ms | 3.32 ms |

(medians of 40 runs in interleaved rounds; ResNet-50 in a process with no
thread on an efficiency core). ResNet-50's convolution groups are a few
percent slower fused than their parts, but so are its convolutions without
an epilogue, whose code is the same in both: the difference comes from
their surroundings, not from fusion.

An epilogue is inlined into the block that stores the tile (the copy out of
the accumulator); when that is impossible it is computed under the tile
loop, and failing that it gets a nest of its own (vectorized, parallel),
without knobs, so that the space does not depend on whether fusion
succeeded.

**Two pitfalls these schedules avoid.** Walking row tiles outside column
strips made every tile of a weight matrix larger than the caches stream the
whole strip again: GPT-2's MLP ran at a third of the speed of a
cache-resident matmul until column strips moved outside. And a packed strip
larger than the stack limit lives behind a pointer, where clang (which
ignores `restrict` on local pointers) cannot prove it does not alias the
accumulator: it unrolled the column loop and left it scalar, so BERT's
3072-deep output projection ran at 7 GFLOPS with the strip packed and 80
without. Strips above the limit are not packed.

## Microkernels

The NEON and SME kernels share a BLIS-style driver (`tessera_pack.h`). For a
tile C[0:m, 0:n] (+)= A[0:m, 0:k] B[0:k, 0:n]:

```
for each block of nc columns:          pack B (kc x nc) into nr-wide panels
  for each block of kc reduction steps:
    for each block of mc rows:         pack A (mc x kc) into mr-high panels
      macro kernel: every mr x nr block of C
```

Packed panels are k-major (for every k, the mr values of A or nr values of B
that the microkernel needs are contiguous) and zero-padded to whole panels,
so the microkernels' loads are sequential and only edge stores need care.
Packing transposes 4 x 4 blocks with NEON `trn` instructions. B can also be
read transposed (`*_sgemm_bt`), for attention's Q·Kᵀ.

Two changes to this order matter for the skinny products of a transformer
(128 rows of activations against a weight matrix), where each element of B
feeds only 256 flops and packing costs about as much as multiplying:

- When A is a single block of rows (m ≤ mc), the column loop runs inside
  the reduction loop, so A is packed once per block of the reduction instead
  of once per block of columns.
- A B operand that is a model constant (`constant_b`, set by the templates
  for graph constants) is packed at its first call, every block in the order
  the driver uses them, and the packing is kept: later calls only read the
  panels. Entries are keyed by the operand's address and geometry, in a
  hash table (a tuned GPT-2 makes 1434 packings and 5796 lookups per run;
  a list took 0.9 µs per lookup at 4000 entries, the table 14 ns), and the
  executor drops them all whenever it loads a model (as `run_kernel` does
  before every call), so a packing is never reused for other data that came
  to live at the same address.

**NEON** (`tessera_neon.h`): an 8 x 12 block of C in 24 of the 32 vector
registers. Per step of k it loads two registers of A and three of B and
issues 24 `fmla` by element (a row of B times one lane of A); 24
independent accumulators cover the FMA latency of four pipes. Blocking: mc
96, kc 256 (an A block of 96 KiB in L2, a B panel of 12 KiB in L1), nc 2040.

**SME** (`tessera_sme.h`): the M4's streaming vectors hold 16 floats and its
ZA array four 16 x 16 fp32 tiles, so a 32 x 32 block of C lives in ZA. Each
step of k loads 32 values of A and 32 of B (one SME2 two-register load each)
and issues four `fmopa` outer products, 1024 multiply-adds. Packing runs
outside streaming mode with NEON; then one `__arm_locally_streaming
__arm_new("za")` call computes every 32 x 32 block of a packed mc x nc
block, so entering and leaving streaming mode (`smstart`/`smstop`, which
save and restore vector state) happens once per macro block of at least
mc·nc·kc/1024 outer products rather than per block of C. Edge blocks load
and store ZA rows with `whilelt` predicates. Blocking: mc 256, nc 512,
kc 1024.

**Convolution as implicit GEMM** (`tsr_neon_conv2d`, `tsr_sme_conv2d`):
output pixels x output channels of an NHWC convolution with HWIO weights is
a GEMM with K = KH·KW·C whose B is the weight tensor itself (HWIO is a
row-major [KH·KW·C, O] matrix). The A operand, the im2col matrix, is never
built: `tsr_pack_im2col` packs the panels it needs straight from the image,
four output pixels at a time, walking runs of contiguous channels per
(kh, kw) and writing zeros where the window leaves the image. Strides,
dilations and asymmetric padding are arguments.

**How templates use them.** Each microkernel is registered with
`tir.intrin.register` as a tensorize intrinsic whose description is made
from the block it replaces (`templates/micro.py`): the group's update block
after `decompose_reduction`, with the loops of the replaced nest rebound to
fresh description loops and the loops outside replaced by 0. Tensorize then
checks the nest against it like any intrinsic, the indices differing only by
the tile's offsets. What the backend adds is the recognition of blocks the
microkernels compute exactly:

- GEMM: `C[.., i, j] += A[.., i, k] * B[.., k, j]` (or `B[.., j, k]`) with
  unit stride along k in A and along j in C. The description gives each
  operand a buffer of its own, so GPT-2's Q and K, two slices of one
  projection, get separate pointers.
- Convolution: `Y[n, oh, ow, o] += X'[n, ih, iw, c] * W[kh, kw, c, o]` where
  `X'` is the image load, alone or under the Select that lower_group writes
  for padding, with `ih = oh·SH + kh·DH − PT` (clamped on the sides that can
  leave the image) and likewise `iw`. The nest covers the whole output plane
  of an image, because a clamped index differs from its tile-relative form
  by more than an offset.

For `c:sme` the same matmul becomes a loop over four 64 x 32 tiles of C, one
per thread: the init zeroes the tile, one call computes it, and the epilogue
runs on it while it is in cache:

```c
static void matmul_parallel0_body(float* restrict x, float* restrict w, float* restrict b, float* restrict relu, float* restrict matmul_1, int64_t begin, int64_t end) {
  for (int64_t n_0_m_0_fused = begin; n_0_m_0_fused < end; ++n_0_m_0_fused) {
    for (int64_t m_1_init = 0; m_1_init < 64; ++m_1_init) {
      #pragma clang loop vectorize(enable) interleave(enable)
      for (int64_t n_1_init = 0; n_1_init < 32; ++n_1_init) {
        {  // matmul_init
          const int64_t v_m_1 = m_1_init;
          const int64_t v_n_1 = n_0_m_0_fused * 32 + n_1_init;
          matmul_1[v_m_1 * 128 + v_n_1] = 0.0f;
        }
      }
    }
    {  // matmul
      tsr_sme_sgemm(64, 32, 256, x, 256, w + n_0_m_0_fused * 32, 128, matmul_1 + n_0_m_0_fused * 32, 128, 1);
    }
    for (int64_t ax0 = 0; ax0 < 64; ++ax0) {
      #pragma clang loop vectorize(enable) interleave(enable)
      for (int64_t ax1 = 0; ax1 < 32; ++ax1) {
        {  // relu
          const int64_t v_i0 = ax0;
          const int64_t v_i1 = n_0_m_0_fused * 32 + ax1;
          relu[v_i0 * 128 + v_i1] = fmaxf(matmul_1[v_i0 * 128 + v_i1] + b[v_i1], 0.0f);
        }
      }
    }
  }
}

static void matmul(float* restrict x, float* restrict w, float* restrict b, float* restrict relu) {
  static _Thread_local float* matmul_1_storage;
  if (!matmul_1_storage) matmul_1_storage = (float*)tsr_alloc(32768);
  float* restrict matmul_1 = matmul_1_storage;
  {
    matmul_parallel0_args args = {x, w, b, relu, matmul_1};
    tsr_parallel_for_n(0, 4, 4, matmul_parallel0, &args);
  }
}
```

Anything else, such as an elementwise prologue inlined into a load or the
NCHW input of ResNet's first layer, keeps the generated loops. The `impl`
builds the call from the match only (pointers, strides, regions); the
intrinsic, named after its description's structural hash, is shared by every
nest with the same structure. The interpreter runs a NumPy reference in
place of the call, so tensorized schedules are checked in the interpreter as
well as natively (`test_c_templates.py`, every ISA the machine has).
Templates parallelize the tile loops before tensorizing, because a
tensorized block passes every operand by address and the parallel check
counts addressed buffers as written.

## Measurements

`uv run python -m tessera.backend.c.bench` writes every table below to
`results/cpu/` as Markdown and JSON (about seven minutes; `--quick` runs a
few small cases, `--only STUDY` one study; the `sme_models` study needs the
exported models). A kernel's time is the median of five samples from
`measure`, each a loop of calls on random inputs, unless a study says it
keeps each variant's best time over interleaved rounds. All numbers were
measured on 2026-11-12 with no other work on the machine; the load averages
the files record (1.8 to 3.5) come from the benchmarks' own threads.

**GEMM** (`gemm.md`, fp32 GFLOPS). One thread: the generated loops 107 to
117 (63 at 2048³ and 23 at 4096³, where a K x 16 strip of B is too large to
pack and every row tile reads it with a stride of N floats), the NEON
microkernel 116 to 123, the SME microkernel 1112 to 1496, NumPy's matmul
(Accelerate) 1412 to 1767. Four threads: the generated loops 290 to 379, NEON
356 to 412, NumPy 1276 to 1861. Accelerate gains little from threads, as
expected of a library that runs on the matrix unit, and on one thread the SME
microkernel comes within 7 to 21 percent of it.

**ResNet-50 convolutions** (`conv.md`, GFLOPS). On four threads the
generated loops reach 200 to 352 and NEON (the microkernel on the deep
layers, the loops on the 64-channel ones) 293 to 375. SME reaches 329 to
1106 on one thread on every layer but the first (whose NCHW input keeps the
generated loops) and less on four threads in these sustained runs.

**Schedule choices** (`schedules.md`, best over interleaved rounds). On the
transformer product with a 3072-wide weight matrix, column strips outside
the row tiles with each strip packed give 113 GFLOPS against 95 with row
tiles outside; where the weight fits the caches the variants are within a
few percent. On NEON with four threads the convolution microkernel beats
the generated loops on layer3's 3 x 3 (353 against 225 GFLOPS) and layer4's
1 x 1 (397 against 229) and loses on layer1's 64-channel 3 x 3 (238 against
369), where its 16-channel strips leave a third of the second 12-wide panel
empty: this is the rule of the conv2d template's NEON default.

**SME and threads.** The four performance cores share one SME unit, and
how much more threads help depends on how long they keep it busy:

- The microkernel alone (`sme_threads.md`, rows of C split between threads):
  at 1024³, 1342 GFLOPS on one thread, 1585 on two and 1624 on four; at
  128 x 3072 x 768, 1103 on one and 595 on four. Giving one thread SME and
  three NEON never beats SME alone (at best 1357 at 1024³ with 90 percent of
  the rows on SME): the NEON part becomes the slowest thread.
- The matmul template's tiles in short interleaved bursts (`sme_template.md`):
  four threads 1576 to 1585 GFLOPS, one thread 962 to 1118, since other
  threads overlap one thread's packing and epilogue with its streaming
  computation.
- In sustained single-kernel runs (`gemm.md`), several threads streaming
  through the unit lose to one on most shapes.
- On whole models, with the thread count forced to one, to four, or chosen
  by the matmul template's rule (every thread when a product has no more
  tiles than threads, one thread otherwise; convolutions on every thread),
  median of 40 runs in interleaved rounds, with the workers sleeping between
  loops:

  | c:sme | default | 1 thread | 4 threads |
  |---|---:|---:|---:|
  | ResNet-50 | 12.0 ms | 25.1 ms | 12.0 ms |
  | BERT | 26.9 ms | 30.9 ms | 26.1 ms |
  | GPT-2 | 37.8 ms | 43.8 ms | 50.1 ms |

  The rule is within 3 percent of the better fixed choice on every model
  and avoids the 33 percent loss of four threads on GPT-2, whose 50257-wide
  output layer has 50 tiles that would keep all four threads contending for
  the unit.

The microkernel and template studies above (and `sme_models.md`, which
gave ResNet-50 18.2 ms by default) ran while idle workers still polled, so
some of their multi-threaded runs had a thread on the efficiency cluster
(see "Threads on SME targets"); the losses of four threads there are partly
that.

## Limitations

- **Kernels are fp32.** The code generator handles the other element types
  (float16 storage, integer indices, boolean masks), but the microkernels and
  the vectorized math functions are single precision only, and there are no
  quantized kernels.
- **Microkernel coverage.** The GEMM microkernel needs unit stride along the
  reduction in A and along the columns in C; the convolution microkernel an
  NHWC image, HWIO weights and `groups == 1`. Depthwise convolutions and the
  first convolution of the image models (its NCHW input is transposed by a
  prologue inlined into the load) use generated loops on every target; a
  matmul operand with a prologue does too unless the prologue is staged
  (above) or copies a strided matrix.
- **Convolution parallelism.** A convolution call covers the whole output
  plane of an image, because a clamped padding index differs from its
  tile-relative form by more than an offset and tensorize matches indices up
  to offsets. Threads share the work by output-channel strips, so a layer
  with 64 output channels has at most four strips of 16.
- **Coarse loops and slow threads.** Threads claim the iterations of a
  loop, but a loop with no more iterations than threads (most SME tiles)
  still waits for its slowest iteration. On SME targets macOS may keep one
  worker on the efficiency cluster for a whole process; that worker then
  skips every loop, and models run on three threads (ResNet-50 untuned:
  15.5 ms instead of 11.8 ms). Thread placement on Linux is left to the
  scheduler.
- **Memory for packed weights.** The microkernels keep a packed copy of
  every constant weight they multiply, as large as the weights themselves,
  for as long as the model is loaded.
- **Packing above 64 KiB.** A packed strip larger than the stack limit is not
  made at all (see Templates), so K x 16 strips of B are packed only up to
  K = 1024.
- **Aliasing.** Kernel parameters are `restrict`; a plan that passes one
  buffer as two parameters of a kernel is undefined (the memory planner never
  does).
- **Portability.** `c:neon` builds with `-mcpu=native`, so its library runs
  on the kind of machine that built it. A `CModel` serializes its runs (one
  arena per model).
- **Math accuracy.** `tanh` and `erf` are within 5 and 6 ulp (below 4e-7
  absolute); ordinary libm calls are not used.
