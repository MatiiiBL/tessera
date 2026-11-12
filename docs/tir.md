# The loop IR: tir, te and compute

Every kernel Tessera generates goes through the loop IR. `compute.lower_group`
turns a fused group of graph operators into an unscheduled `PrimFunc`; a
backend template rewrites its loops with the schedule primitives of
`tir.Schedule`; the C and Metal code generators print the result. This
document describes the IR, the analyses around it, the interpreter that
checks schedules, the tensor-expression layer, the compute definitions of
the operators, every schedule primitive, and how the claim "schedules never
change the result" is tested.

| Package | Contents |
|---|---|
| `tessera.tir` | IR nodes (`expr.py`, `stmt.py`), visitors and substitution (`functor.py`), printer (`printer.py`), simplifier and bounds (`arith.py`), verification (`verify.py`), region inference and structural hashing (`analysis.py`), interpreter (`interpreter.py`), intrinsics (`intrin.py`), schedule (`schedule.py`, `primitives/`), lowering passes (`transform.py`), interpreter benchmark (`bench.py`) |
| `tessera.te` | placeholders, compute stages, reductions, `create_prim_func` |
| `tessera.compute` | the compute definition of every operator, `lower_group` |

## 1. The IR

The IR follows TensorIR (TVM). A `PrimFunc` has buffer parameters (inputs,
then outputs) and a body that is a *root block*, which allocates the
function's intermediate buffers. Inside are loops (`For`) and *blocks*. A
block is a unit of computation: it declares iteration variables, each with a
domain and a kind (spatial or reduce), the buffer regions it reads and
writes, an optional `init` for reductions and a body of stores. A
`BlockRealize` binds the block's iteration variables to expressions of the
loops around it and carries a predicate.

This is a 4x4 matmul followed by a ReLU, as `te.create_prim_func` builds it
and `tir.script` prints it:

```
@prim_func
def matmul_relu(A: float32[4, 4], B: float32[4, 4], D: float32[4, 4]):
    C = alloc_buffer(float32[4, 4])
    for i in range(4):
        for j in range(4):
            for k in range(4):
                with block('C'):
                    v_i = spatial(4, i)
                    v_j = spatial(4, j)
                    v_k = reduce(4, k)
                    reads(A[v_i, v_k], B[v_k, v_j])
                    writes(C[v_i, v_j])
                    with init():
                        C[v_i, v_j] = 0.0
                    C[v_i, v_j] = C[v_i, v_j] + A[v_i, v_k] * B[v_k, v_j]
    for i_1 in range(4):
        for j_1 in range(4):
            with block('D'):
                v_i_1 = spatial(4, i_1)
                v_j_1 = spatial(4, j_1)
                reads(C[v_i_1, v_j_1])
                writes(D[v_i_1, v_j_1])
                D[v_i_1, v_j_1] = max(C[v_i_1, v_j_1], 0.0)
```

The rules that give this form its meaning:

- **Blocks are isolated.** The body of a block with iteration variables
  refers only to those variables, never to the loops around it; the loops
  reach it only through the bindings. Schedule primitives rewrite loops and
  bindings and leave block bodies alone, so a block always computes the same
  thing at the same point of its domain.
- **A block runs only where its predicate holds.** Imperfect splits add
  predicates such as `i_0 * 3 + i_1 < 8`.
- **A reduction's init runs when every reduce variable is at the start of
  its domain**, before the first update of that output element.
- **Variables compare by identity**, so loops are named by their `Var`. The
  printer gives a suffix to a second variable with a name already used
  (`i_1` above).
- Expressions are immutable; `Select` may evaluate both sides, so a load
  inside a select must be in bounds by itself (padding clamps its indices).

`IterVar` doubles as its variable in index arithmetic (`x[h + kh]` with a
reduction axis `kh`), which is what makes `te` bodies read naturally.

### Analyses

- `tir.functor`: `StmtVisitor`, `StmtMutator` (returns unchanged subtrees
  by identity, which the schedule relies on to find loops) and `substitute`
  for variables and buffers.
- `tir.script(func)`: the text form above. `tir.structural_hash(func)` hashes
  the same text with every name replaced by its order of appearance, so two
  groups that differ only in names hash the same
  (`tests/tir/test_ir_tools.py`).
- `tir.simplify` / `tir.Analyzer`: integer expressions are put in a
  canonical linear form `c0 + c1*t1 + ...` with like terms combined.
  `floordiv` and `floormod` by constants pull out multiples of the divisor
  and use the operands' bounds (`(i*4 + j) // 4 -> i` for `j` in `[0, 4)`),
  split off small remainders (`(2*a + b) // 4 -> a // 2` for `b` in
  `[0, 2)`) and recombine the digits of a fused index
  (`f // 4 * 4 + f % 4 -> f`, and three-level digits likewise).
  Comparisons are normalized and decided from bounds, which is how split
  predicates disappear when they always hold. Float expressions are only
  constant-folded; they are never reassociated. `Analyzer.bound` gives a
  constant interval; `Analyzer.relax` gives symbolic bounds where some
  variables range over their intervals and the others stay symbols
  (`i*4 + j` with `j` relaxed is `[i*4, i*4 + 3]`), which is region inference.
  Hypothesis tests compare the simplified form with the original at every
  point of the variables' ranges (`tests/tir/test_arith.py`).
- `tir.verify(func)`: scoping (each variable defined once and used inside its
  scope, block bodies isolated), bindings inside their domains (taking the
  predicate into account, by interval reasoning and, when that fails, by
  enumerating up to 16,384 combinations of the variables involved), every
  load, store and declared region inside its buffer (narrowed under `if`
  conditions), buffers that are parameters or allocated by an enclosing
  block, operand types, unique block names, known thread axes, and the
  per-thread rule for local buffers (section 3).

## 2. The interpreter

`tir.interpret(func, arrays)` runs a function on NumPy arrays and writes the
outputs in place. It does not walk the tree per element: it translates the
function once into Python source (`tir.python_source(func)` shows it) and
compiles it. Loops become `for` loops, blocks become their bindings, an `if`
for the predicate and an `if` for the init, and loads become indexing into
flat `array.array` storage with the row-major offset simplified once at
translation time. A read-modify-write of an `array.array` element costs about
a third of the same operation on a NumPy array (table below); a Python list
is cheaper still, but NumPy can view an `array.array` without a copy, which
is how intrinsic references see the buffers. Nests deeper than 15 loops are
outlined into separate functions (Python allows 20 nested blocks). The
compiled kernel is cached on the function object.

Semantics follow the C backend: `floordiv`/`floormod` round toward negative
infinity, integer `div` truncates, float arithmetic follows IEEE (division by
zero gives infinities, `exp` overflows to infinity, `min`/`max` propagate
NaN), values are rounded to float32 on every store, integer stores wrap.
Buffers the function allocates start as NaN (floats), so reading an element
nothing wrote shows up in the output; `check_bounds=True` checks every index.

Speed, from `results/tir/interpreter.md` (regenerate with
`uv run python -m tessera.tir.bench`). These numbers were taken on
2026-11-12 while the machine ran other work (1-minute load average 6.8), so
they are provisional:

| kernel | block executions | compile (ms) | run (ms) | ns per execution |
|---|---:|---:|---:|---:|
| matmul 64x64x64 | 262,144 | 0.6 | 81.8 | 312 |
| matmul 64x64x64, tiled 8x8 | 266,240 | 0.8 | 99.1 | 372 |
| conv2d 3x3 NHWC 16x16x16 -> 16, relu | 593,920 | 1.5 | 380.0 | 640 |
| softmax 64x512 | 98,304 | 0.8 | 22.3 | 227 |
| gelu 256x256 | 65,536 | 0.3 | 15.2 | 233 |

| storage (one `x[5] = x[5] + x[7] * x[8]`) | ns |
|---|---:|
| array.array | 100 |
| list | 52 |
| numpy | 302 |

Translating and compiling a kernel takes a millisecond or two. At a few
hundred nanoseconds per block execution, kernels of a few million iterations
run in about a second, which bounds the sizes the tests interpret; the
convolution costs more per execution because its padded loads clamp and
select.

## 3. Threads and local buffers

Thread-bound loops run one after the other in the interpreter, which
matches Metal for the IR the schedule produces, with one rule taken from
Metal: a `local` buffer allocated outside the `thread.*` loops belongs to
each thread, so the interpreter keeps one copy of it per thread index. That
makes the standard GPU accumulator valid: the init, update and copy-out
nests of a `cache_write(..., "local")` buffer each bind `thread.y` and
`thread.x`, and every thread sees its own register tile
(`test_gpu_matmul_with_a_per_thread_accumulator`). `tir.verify` and the
interpreter reject a local buffer used under different sets of thread axes
(for example inside a thread loop and outside it), or indexed by thread in
different ways (a nest in which each thread writes a row and one in which it
reads a column), since a thread could then read elements another thread
wrote (`test_thread_loops_must_index_a_local_buffer_the_same_way`). The
check compares, for every use, the terms of each index that involve a thread
loop, with the loops named by their axis. It is not run after each
primitive, because a schedule passes through such states while it binds its
nests one at a time.

## 4. Tensor expressions

```python
A = te.placeholder((M, K), DType.float32, "A")
B = te.placeholder((K, N), DType.float32, "B")
k = te.reduce_axis(K, "k")
C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), "C")
func = te.create_prim_func([A, B, C])
```

Each stage becomes one block, with one loop per axis (spatial axes, then
reduction axes) and identity bindings, under the root block, which allocates
every stage that is not a parameter. `te.sum`/`te.max`/`te.min` give the
block an init with the combiner's identity; `te.compute(..., init=fn)`
replaces it with an expression of the spatial axes (a convolution starts from
its bias), and `annotations=` are attached to the block.

## 5. Compute definitions and lower_group

`compute.lower_group(group)` builds a `LoweredGroup` for every operator of
design section 3. Elementwise, broadcast and injective operators are
*lazy tensors*: a function from indices to an element expression. When an
anchor or reduction reads a lazy tensor, the prologue is inlined into its
loads; the epilogue chain composes into one expression stored by one block.
So a fused group has a main block (the anchor), at most one epilogue block,
and for Softmax and LayerNorm the statistics blocks the main block reads.
Every block carries a `role` annotation:

| group | blocks (role) | `dims` |
|---|---|---|
| Conv2d (NHWC, NCHW; strides, pads, dilations, groups, depthwise) | pad (with `pad_block`), main, epilogue | N H W C KH KW O OH OW SH SW DH DW G PT PL PB PR PAD_BLOCK |
| MatMul (rank >= 2, broadcast batch) | main, epilogue | B M N K |
| MaxPool2d, AvgPool2d | main, epilogue (AvgPool divides there) | N H W C KH KW OH OW SH SW |
| ReduceSum, ReduceMax, ReduceMean | main, epilogue (ReduceMean divides there) | OUTER R INNER |
| Softmax | `<main>_max` (max), `<main>_sum` (sum), main, epilogue | OUTER R INNER |
| LayerNorm | `<main>_mean` (mean), `<main>_var` (var), main, epilogue | OUTER R INNER |
| elementwise, broadcast, injective only | one block (injective); `main_block` is None | SIZE |

The main block's loops are the output axes in order, then the reduction axes
(convolution: `kh`, `kw`, `ic`; `ic` is left out when a group has one input
channel, as in a depthwise convolution). A convolution's bias is the
reduction's init. Padding is a select whose loads clamp their indices, and
only the sides that can leave the input get a test. A Gather index is read
at run time, wrapped if negative and clamped into range. A dimension that is
1 in both an operand and the output keeps the output's iteration variable,
so batch-1 epilogues and prologues read by iteration variable and the
inlining primitives accept them.

`lower_group(group, pad_block=True)` (what the Metal backend asks for through
`Backend.lowering_options`) moves a padded convolution's select out of the
main block: the padded input becomes a block of its own, `<main>_pad` with
role `pad`, covering only the rows and columns the windows read
(`(OH - 1) * SH + (KH - 1) * DH + 1` rows, likewise for columns), and the main
block's update is `out += pad[n, oh * SH + kh * DH, ow * SW + kw * DW, ic] *
w[...]`, a plain multiply-accumulate that a GEMM-like microkernel description
matches. `dims["PAD_BLOCK"]` says whether the block exists: unpadded
convolutions get none in either mode, and inlining the pad block
(`compute_inline`) gives back the default function exactly
(`test_conv2d` checks both modes and the round trip over its Hypothesis
examples; `test_padded_conv_staged_through_shared_memory` computes the pad
block per tile of output rows, stages it in threadgroup memory and binds the
threads).

Each operator is tested by interpreting its lowered group against the
operator references in float64 (`tests/compute/test_lower.py`), the
convolution with Hypothesis over sizes, strides, pads, dilations, groups,
bias and layout, plus a fused small network and the groups of
`passes.optimize` on an NCHW model. With `-m models`,
`tests/compute/test_models.py` lowers every group of the benchmark models
(54 in MobileNetV2, 134 in BERT, 111 in GPT-2), verifies it, applies a
CPU-style and a GPU-style default schedule (`tests/compute/schedules.py`),
and interprets the groups of up to two million loop iterations (16, 38 and
38 of them) against the references.

## 6. Schedule primitives

`tir.Schedule(func)` holds the current function; every primitive either
rewrites it or raises `ScheduleError` and leaves it unchanged, and appends
a line to `trace`. Loops are named by their `Var`, blocks by name. The
snippets below show the loops and block bodies, without the `reads`/`writes`
lines.

### split, fuse, reorder

`split(loop, factors)` replaces a loop by nested loops, outermost first; one
factor may be `None` (inferred). An imperfect split adds a predicate to the
blocks underneath. Splitting `i` (extent 8) by 3:

```
for i_0 in range(3):
    for i_1 in range(3):
        for j in range(8):
            for k in range(4):
                with block('C'):
                    v_i = spatial(8, i_0 * 3 + i_1)
                    ...
                    where(i_0 * 3 + i_1 < 8)
```

`fuse(*loops)` merges perfectly nested consecutive loops; the bindings get
the digits of the fused index:

```
for i_j_fused in range(16):
    with block('D'):
        v_i_1 = spatial(4, i_j_fused // 4)
        v_j_1 = spatial(4, i_j_fused % 4)
```

`reorder(*loops)` permutes loops of one perfect nest (`reorder(k, i)` turns
`i, j, k` into `k, j, i`). It refuses loops separated by other statements.

### Loop kinds

`parallel`, `vectorize` (innermost loops only), `unroll`, `bind(loop, tag)`
(a Metal grid axis) and `annotate(loop or block, key, value)`:

```
for i_j_fused_0 in thread_binding(8, thread='threadgroup.x'):
    for i_j_fused_1 in thread_binding(32, thread='thread.x'):
        with block('D'):
            v_i_1 = spatial(10, (i_j_fused_0 * 32 + i_j_fused_1) // 24)
            v_j_1 = spatial(24, (i_j_fused_0 * 32 + i_j_fused_1) % 24)
            where(i_j_fused_0 * 32 + i_j_fused_1 < 240)
```

### compute_inline, reverse_compute_inline

`compute_inline(block)` substitutes a spatial producer into its consumers;
`reverse_compute_inline(block)` moves a consumer into its only producer. On
`Y = X * 2; Z = Y + 1`, either one gives a single block storing
`Z[v_i, v_j] = X[v_i, v_j] * 2.0 + 1.0`.

### compute_at, reverse_compute_at

`compute_at(block, loop)` moves a producer under a loop of its consumers and
computes only the region they read there; `reverse_compute_at(block, loop)`
moves a consumer under a loop of its producer. Regions come from relaxing
the bindings over the loops inside the target (`Analyzer.relax`); a
predicate keeps partial tiles inside the domain. After
`split(i, [None, 2])` on `Z` and `compute_at('Y', i_0)`:

```
for i_0 in range(3):
    for ax0 in range(2):
        for ax1 in range(4):
            with block('Y'):
                v_i = spatial(6, i_0 * 2 + ax0)
                v_j = spatial(4, ax1)
                Y[v_i, v_j] = X[v_i, v_j] * 2.0
    for i_1 in range(2):
        for j in range(4):
            with block('Z'):
                v_i_1 = spatial(6, i_0 * 2 + i_1)
                ...
```

A stencil's producer gets the halo (`test_compute_at_infers_halo_region`).
`reverse_compute_at` additionally checks that the producer writes exactly the
tile in each iteration: by counting when the predicate is trivial, by
enumerating every iteration when there are at most 2^20, and otherwise by
checking that the predicate only clips indices at the buffer's edge (GPT-2's
50257-wide output tiled by 4).

### cache_read, cache_write, decompose_reduction

`cache_read(block, i, scope)` copies the buffer behind the block's i-th read
region into a new buffer and makes the block read it; `cache_write(block, i,
scope)` makes the block write a new buffer and copies it out. Both keep the
buffer's full shape and return the copy block, which is then placed with
compute_at or reverse_compute_at (`compact_buffers`, section 8, shrinks it).
A register tile:

```
C_local = alloc_buffer(float32[4, 4], 'local')
for i in range(4):
    for j in range(4):
        for k in range(4):
            with block('C'):
                ...
                C_local[v_i, v_j] = C_local[v_i, v_j] + A[v_i, v_k] * B[v_k, v_j]
    for ax1 in range(4):
        with block('C_local'):
            v0 = spatial(4, i)
            v1 = spatial(4, ax1)
            C[v0, v1] = C_local[v0, v1]
```

`decompose_reduction(block, loop)` moves the init into its own block before
`loop`, over copies of the spatial loops at and inside it:

```
for i in range(4):
    for j in range(4):
        with block('C_init'):
            ...
            C[v_i_1, v_j_1] = 0.0
        for k in range(4):
            with block('C'):
                ...
                C[v_i, v_j] = C[v_i, v_j] + A[v_i, v_k] * B[v_k, v_j]
```

### When primitives refuse

Each primitive checks the conditions under which the result cannot change:

- **Reductions.** Every loop around a reduction block is classified by what
  its variable feeds: spatial bindings, reduce bindings, both, or neither. A
  loop may not feed both (fusing a spatial loop with a reduction loop would
  let a later reorder visit an output element first at a non-initial
  reduction step), and a loop the block does not use may not sit inside the
  block's reduction loops (it would repeat the reduction or re-run the
  init). Blocks whose init was split off cannot be moved.
- **Concurrency.** `parallel`, `vectorize` and `bind` require that different
  iterations write different elements or private buffers (all accesses
  inside the loop). Distinctness is proved by a mixed-radix test on the
  row-major position of the written point (fused and split loops pass it
  even when the split is imperfect), by a box count, or by enumerating up to
  2^20 iterations per iteration of the enclosing loops. A loop that drives a
  reduction is refused, except for `thread.*` bindings, which are
  cross-thread reductions the Metal backend lowers cooperatively.
- **Order.** Moving or inlining a block keeps its producers before it and
  its consumers after it: no block in between may write its inputs or touch
  its output, all consumers must be under the target loop for compute_at,
  and a producer must have finished its reduction within one iteration of
  the target loop for reverse_compute_at.
- **Form.** Inlined producers store to distinct spatial variables over the
  whole buffer (a constant 0 on an extent-1 dimension counts as that
  dimension's variable); reverse-inlined consumers read their producer
  elementwise.

## 7. Tensorize

A backend registers an intrinsic with a description, an implementation and
a Python reference:

```python
tir.intrin.register("c.sgemm_8x8", tir.intrin.matmul_desc(8, 8), impl, reference)
```

The description is a small PrimFunc of what the intrinsic computes; loop
extents may be variables (`matmul_desc` leaves `K` open).
`Schedule.tensorize(loop, name)` matches the perfect nest at `loop` against
it: loops one to one, block bodies node for node, buffers by
correspondence, indices up to offsets that do not depend on the matched
loops. It then calls `impl(match)` with an `IntrinMatch`: for each
description parameter the matched buffer, the address of the region's first
element and the buffer's strides, plus the matched extents and the values of
the variable extents. `impl` returns the replacement, normally
`Evaluate(Call("c.sgemm_8x8", ...))`. On a matmul tiled by 4:

```
with block('matmul'):
    reads(A[i_0 * 4:i_0 * 4 + 4, 0:6], B[0:6, j_0 * 4:j_0 * 4 + 4],
          matmul[i_0 * 4:i_0 * 4 + 4, j_0 * 4:j_0 * 4 + 4])
    writes(matmul[i_0 * 4:i_0 * 4 + 4, j_0 * 4:j_0 * 4 + 4])
    annotate({'tensorize': 'test.gemm_4x4', 'tensorize.extents': '4,4,6'})
    test.sgemm_4x4(&A[i_0 * 4, 0], &B[0, j_0 * 4], &matmul[i_0 * 4, j_0 * 4], 6, 6, 8, 8)
```

The tensorized block keeps the matched regions, and the interpreter runs
`reference(*views, *extents)` on NumPy views of them, so a scheduled function
with microkernels is checked like any other (`tests/tir/test_tensorize.py`).
The description must match literally: `A[i, k] * B[k, j]` does not match
`B[k, j] * A[i, k]`, and a block with an init matches only a description
with one (decompose the reduction first).

**Edge tiles.** When a spatial loop's extent is not a multiple of the tile,
the last tile carries the split's predicate, for example
`where(j_0 * 64 + j_1 < 50257)` for GPT-2's output projection tiled by 64.
Tensorize accepts a predicate whose conjuncts each bound one matched spatial
loop from above (`A + k * x < c` with `A` free of the matched loops) and
passes `IntrinMatch.valid`: the number of valid iterations of each matched
loop, an expression of the outer loops (`min(50257 - j_0 * 64, 64)`, or the
extent itself where nothing is cut). Conjuncts over the outer loops alone
stay on the tensorized block. A predicate that limits a reduction loop,
bounds a loop from below, relates two matched loops or bounds a loop that
indexes a buffer other than directly is refused. On an edge tile the block's
regions, and `IntrinMatch.regions`, are cut to the valid part (the pointers
still address the tile's first element), and the interpreter applies the
reference to views of that shape; `extents` stays the full tile.

An implementation is expected to compute the whole tile and store only the
valid part. Tensorize leaves the reads to it: a full-tile kernel may read up
to one tile past the end of an input, and keeping those reads inside
allocated memory (the executors pad their allocations) is the backend's
concern, not the IR's (`test_tensorize_edge_tiles_of_imperfect_splits`,
`test_tensorize_the_edge_of_a_wide_output`,
`test_tensorize_refuses_other_predicates`).

## 8. Lowering passes

`tir.lower_blocks(func)` makes every block opaque: bindings are substituted,
inits become `if` statements testing that the reduce variables are at their
start, predicates become `if`s. `tir.compact_buffers(func)` then allocates
every local and shared buffer at the innermost loop containing all its
accesses (shared buffers stay outside `thread.*` loops, because the threads
of a threadgroup share them) with the shape of the region touched there, and
shifts its indices; buffers passed to intrinsics by address keep their shape,
since their strides are part of the call. The tiled matmul's accumulator
becomes a `float32[4, 4]` allocated under the tile loop
(`tests/tir/test_transform.py`).

## 9. How semantics preservation is tested

- Every primitive has unit tests of what it does and when it refuses, each
  interpreting the result against NumPy (`tests/tir/test_schedule.py`).
- `tests/tir/test_schedule_random.py` builds one of four computations (a
  matmul with a bias and ReLU, a padded box filter, a row softmax, an
  elementwise chain with a transpose) with random sizes, applies up to ten
  random primitives with random arguments drawn from the current schedule,
  and checks that every refusal leaves the schedule unchanged, every applied
  step verifies, and the final function interprets to the unscheduled result
  (300 examples per run).
- Hypothesis checks the simplifier, bounds and relaxation pointwise, the
  convolution definition over its attributes, and concurrent loops after
  fuse and imperfect splits over random shapes and factors.
- The model sweep (section 5) schedules every group of the real models.

The package has 189 tests (`uv run pytest tests/tir tests/te tests/compute`,
a few seconds) plus the model sweep (`uv run pytest -m models tests/compute`).
Floating-point results are compared with a tolerance relative to the largest
magnitude, because reordered reductions round differently.

## 10. Design decisions

- **Blocks rather than plain loops.** Isolated blocks with explicit domains
  make the legality checks local: a primitive reasons about bindings and
  regions, never about the arithmetic inside a block. The cost is that code
  generators handle bindings, predicates and inits, which `lower_blocks`
  removes when they do not want to.
- **Full-shape caches, compacted at the end.** cache_read and cache_write
  keep the buffer's shape, so compute_at can place the copy anywhere without
  rewriting indices; `compact_buffers` computes the real footprint once the
  schedule is final.
- **A Python interpreter, not NumPy vectorization.** Generated scalar code
  runs any IR the schedule can produce, including predicates, partial tiles,
  intrinsics and per-thread buffers, with one code path. Its cost of a few
  hundred nanoseconds per block execution limits interpreted tests to small
  sizes.
- **Exact checks where symbolic reasoning runs out.** Binding bounds under
  predicates, concurrency and tile exactness fall back to enumeration over
  small iteration spaces instead of growing the simplifier for every pattern.

## 11. Limitations

- Shapes are static: every loop extent is a constant (descriptions of
  intrinsics excepted).
- The simplifier is not complete. Some fused-and-split index forms are not
  recombined; the concurrency and tile checks then enumerate (up to 2^20
  iterations) and refuse beyond that, and the binding check in `verify`
  accepts predicated bindings it can neither prove nor enumerate within
  16,384 combinations.
- `reorder` works only within a perfect nest; `compute_at` and
  `reverse_compute_at` move one block at a time and need a single writer of
  the moved buffer; `cache_read` refuses a buffer written inside the same
  loop nest as its reader.
- Tensorize matches the description's expression literally (no
  commutativity or reassociation), and on edge tiles accepts only upper
  bounds of matched spatial loops.
- There is no `rfactor`; cross-thread reductions on Metal go through binding
  a reduction loop to a `thread.*` axis.
- The interpreter's speed (section 2) limits interpreted tests to kernels of a
  few million iterations; the model sweep interprets only the smaller groups.
