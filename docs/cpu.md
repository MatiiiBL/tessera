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
- `measure(func, target)` and `prebuild(func, target)`: timing of one
  kernel in a separate process, for the tuner.

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
  allocation (`m_0_n_0_fused >> 3`) are constant while the buffer lives and
  drop out of the shrunk index. Buffers above 64 KiB go on the heap (the
  calling thread runs the first chunk of every parallel loop, and Python
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
are different libraries). `tsr_parallel_for(begin, end, task, args)` splits
the range into one contiguous chunk per thread (static partitioning: the
templates parallelize loops of equal-cost tiles) and the calling thread runs
the first chunk. A loop is published by writing the job and incrementing a
generation counter; workers poll the counter for 0.5 ms (a model's kernels
follow each other within microseconds, so workers stay awake through a run)
and then sleep on a condition variable. The publisher takes the mutex only
when some worker sleeps; the sleeper registers itself before rechecking the
counter under the mutex, so a wakeup cannot be lost. Every worker
acknowledges every job, so the job fields are never rewritten while a
worker may still read them. `tsr_parallel_for_n` limits a loop to fewer
threads (the others acknowledge and stay idle). On macOS the workers ask for
`QOS_CLASS_USER_INTERACTIVE`, which keeps them on the performance cores (the
default thread count is the number of performance cores), and so does every
exported entry point for its calling thread, once: the caller runs the
serial parts and the first chunk of every loop, and a kernel that never
starts the pool (single-threaded SME) otherwise ran at the caller's default
class, which macOS may move to an efficiency core. A second
thread starting a loop while the pool is busy runs it serially, and after
`fork()` the child runs every loop serially. `test_thread_pool` checks
2000 loops of varied sizes for exactly-once execution, wakeups after
sleeping, nesting, concurrent callers and fork.

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
sees) for a dozen FMAs and one division: `erf` costs 0.44 ns per element
against 2.3 ns for the exp-based version it replaced. 2^n is applied by an
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
the median is returned. The harness runs as a subprocess, so a candidate
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
| `matmul`, NEON/SME | tiles of C (edge tiles allowed) computed by the GEMM microkernel, the reduction's init before each call and the epilogue after it | `kernel` (`micro`), `micro_m` (≤ 256 rows), `micro_n` (a whole number of tiles per thread, ≤ 256 columns), `micro_threads` (all) |
| `conv2d` | as matmul with output columns as rows: `tile_ow` columns x `tile_c` channels in registers over KH·KW·C; channel strips outside pixel tiles, each strip's weights packed when they fit the stack | `tile_ow` (4), `tile_c` (16), `order` (`co`), `pack_w` (on) |
| `conv2d`, NEON/SME | strips of output channels computed by the implicit-GEMM microkernel over the whole output plane, bias (the reduction's init) before the call, epilogue after | `kernel` (`micro`), `micro_c` (a whole number of strips per thread, ≤ 256 channels), `micro_threads` (all) |
| `depthwise_conv2d` | `tile_ow` columns x `tile_c` channels accumulated over the window | `tile_ow` (4), `tile_c` (16) |
| `pool` | channels vectorized, the window reduced in a local buffer, the average's scaling on the stored tile | `tile_ow` (1), `tile_c` (16) |
| `reduce` | over the last axes: one accumulator per output element, the contiguous loop vectorized by clang; over middle axes (a global average pool): a `tile` of the innermost output axis accumulated while the reduction streams over contiguous rows | `tile` (the whole axis, or one tile per thread when the outer axes are too few) |
| `softmax`, `layernorm` | every statistics stage computed per row inside the final stage's row loop, reductions decomposed so clang vectorizes them, the final stage's row loop vectorized | `vector` (16) |
| `injective` | stages inlined into the output's, its innermost loop vectorized | `vector` (16) |

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
`results/cpu/` as Markdown and JSON (about six minutes; `--quick` runs a few
small cases, `--only STUDY` one study). Each kernel is timed by `measure`:
the median of five samples on random inputs, each sample a loop of calls.

**These numbers are provisional.** They were measured on 2026-11-12 while
other work was running on the machine (load averages 4 to 7, including
benchmarks that use Accelerate and the SME/AMX unit), which depresses the
multi-threaded results most. They will be replaced by a run on a quiet
machine with the same command.

**GEMM** (`results/cpu/gemm.md`, GFLOPS, fp32): on one thread the generated
loops reach 60 to 100 GFLOPS, the NEON microkernel 75 to 100 and the SME
microkernel 490 to 920; NumPy's matmul (Accelerate) reaches 890 to 1470 and
does not change between one and four threads, as expected of a library that
runs on the matrix unit. The generated loops fall to 18 GFLOPS at 4096 x
4096 x 4096, where a 4096 x 16 strip of B is too large to pack and every row
tile reads it with a 16 KiB stride; the microkernels block for the caches
themselves and keep their speed.

**ResNet-50 convolutions** (`results/cpu/conv.md`): on one thread the SME
implicit GEMM reaches 200 to 560 GFLOPS on every layer but the first (whose
NCHW input keeps the generated loops), against 30 to 70 for the generated
loops and 45 to 60 for NEON.

**Schedule choices** (`results/cpu/schedules.md`, each variant's best time
over interleaved rounds): on the transformer product with a 3072-wide weight
matrix, walking column strips outside the row tiles and packing each strip
gives 113 GFLOPS against 94 with row tiles outside; where the weight fits
the caches the variants are within a few percent. On NEON with four threads
the convolution microkernel beats the generated loops on layer3's 3 x 3
convolution (230 against 168 GFLOPS) and loses on layer1's 64-channel one
(240 against 367), where its 16-channel strips leave a third of the second
12-wide panel empty; this is the rule the conv2d template's NEON default
follows.

**SME and threads** (`results/cpu/sme_threads.md`,
`results/cpu/sme_template.md`): the four performance cores share one SME
unit. The SME GEMM microkernel on its own gains little from more threads
(1324 GFLOPS on one, 1444 on two, 1338 on four at 1024 x 1024 x 1024), and
splitting the rows between one SME thread and three NEON threads is slower
than SME alone at every split from 70 to 95 percent: the NEON part becomes
the slowest thread. The matmul template's SME path behaves differently,
because every tile also packs its operands and runs its init and copy on
NEON, which other threads overlap with one thread's streaming computation:
on the transformer products four threads reach 760 to 890 GFLOPS against
630 to 680 on two and 440 to 470 on one. The templates therefore default to
every thread. In the median-based tables (`gemm.md`, `conv.md`), measured
while other work loaded the machine, SME on four threads comes out below one
thread: a median counts the samples in which other processes took cores,
which multi-threaded kernels feel most. The quiet re-run settles this.

## Limitations

- **Kernels are fp32.** The code generator handles the other element types
  (float16 storage, integer indices, boolean masks), but the microkernels and
  the vectorized math functions are single precision only, and there are no
  quantized kernels.
- **Microkernel coverage.** The GEMM microkernel needs unit stride along the
  reduction in A and along the columns in C; the convolution microkernel an
  NHWC image, HWIO weights and `groups == 1`. Depthwise convolutions, the
  first convolution of the image models (its NCHW input is transposed by a
  prologue inlined into the load) and products with an elementwise prologue
  use generated loops on every target.
- **Convolution parallelism.** A convolution call covers the whole output
  plane of an image, because a clamped padding index differs from its
  tile-relative form by more than an offset and tensorize matches indices up
  to offsets. Threads share the work by output-channel strips, so a layer
  with 64 output channels has at most four strips of 16.
- **Static partitioning.** The pool gives each thread one equal chunk of a
  parallel loop; with no work stealing, a thread delayed by other work on the
  machine delays the whole loop. Thread placement on Linux is left to the
  scheduler.
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
