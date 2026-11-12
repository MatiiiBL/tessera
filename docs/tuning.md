# Tuning

Every schedule template in Tessera asks a `Config` for each decision it
makes: how to tile a loop, whether to unroll or vectorize it, how many
threads a Metal threadgroup gets. `tessera.tune` turns those decisions
into a search problem, measures candidates on the machine, learns a cost
model from the measurements while it searches, and writes everything to a
log that `tessera.compile` reads. This document describes how each piece
works and why it is built that way. The results are in
[results/tuning.md](../results/tuning.md).

## Knobs and spaces

A template is written once against `Config` (`tessera/tune/space.py`):

```python
def schedule_matmul(sch, lowered, cfg):
    mo, mi = sch.split(m, cfg.split("tile_m", M, parts=2))
    if cfg.choice("unroll_k", [False, True]):
        ...
```

Run with a `DefineConfig`, the template records every knob and its
options and gets each knob's default; `space_of` does this once to build
the `ConfigSpace`, the product of all options. Run with an `ApplyConfig`,
the template gets the values of one point. `split` knobs list every
factorization of an extent (or, with `policy="power2"`, power-of-two inner
factors with an imperfect outer loop), sorted so that neighbouring options
have the same innermost factor; the search uses that order.

## Tasks

A task (`tessera/tune/task.py`) is one lowered group, the template for its
pattern and the target. Its key is `<pattern>:<structural hash>`, where the
hash of the unscheduled `PrimFunc` ignores every name, so the identical
3×3 convolutions repeated through a ResNet stage are one task. `extract_tasks` merges
groups with equal keys and counts them in `weight`. The target is part of
each log record (`Target.key` includes the ISA, the thread count and the
CPU name), so a log tuned on one machine is never applied to another.

## Measuring candidates

`LocalRunner` (`tessera/tune/measure.py`) handles a batch of candidates in
two phases:

1. **Build** in up to three processes at once. A build applies the
   template to the configuration and, when the backend is a `Prebuilder`
   (both backends are), compiles the kernel into the build cache. Three is
   the limit because the machine is shared and clang is the expensive part.
2. **Measure** one candidate at a time in a separate process, through
   `backend.measure`, with no builds running, so candidates do not compete
   for cores or memory bandwidth while they are timed.

Each process is a fresh interpreter (`python -m tessera.tune.measure`)
that talks to the tuner over two pipes. A candidate that raises, crashes
(a segmentation fault in generated code) or runs past its timeout (120 s
for a build, 60 s for a measurement) becomes a failed result with a
reason, its process is replaced, and the batch continues. Processes stay
alive between batches and receive each task once, so a task with large
constants is not sent again for every candidate. `multiprocessing`'s
`spawn` was the first implementation; it re-runs the caller's main script
in every child, which breaks any script without a `__main__` guard, so the
workers now start their own interpreter.

## Tuning logs

A log is JSON lines, one measurement per line, exactly as in the design:

```json
{"target": "c-sme-t4-Apple_M4", "task": "matmul:3f2a...", "config": {"tile_m": [8, 16]},
 "seconds": 0.000412, "error": null, "time": "2026-11-12T10:00:00Z"}
```

Records are appended and flushed as each measurement finishes, so an
interrupted run loses at most the candidate in flight. A run reads the log
first, skips every configuration it already holds (failures included) and
trains the cost model on them, so running the same command again resumes
it. Lines that do not parse, such as one cut off by a killed process, are
ignored. `BestConfigs` loads any number of log files and directories and
returns the fastest successful record of a (target, task) pair that the
current template still accepts, so a log written before a template gained
or renamed a knob falls back to the next valid record or to the defaults.

## Search

Both strategies measure the template's default first, so every curve
starts from the untuned schedule, and both never measure a configuration
twice.

**Random search** samples unmeasured points uniformly.

**Model-guided evolutionary search** (`tessera/tune/search.py`) follows
the idea of Ansor. Each round:

1. Start a population of 128: the 16 fastest measured configurations and
   random points.
2. Evolve it for 4 generations. The best 20% survive unchanged; each other
   child comes from a tournament between two parents, takes each knob from
   one of two parents with probability 0.5 (crossover), and changes at
   least one knob: half the time to a neighbouring option (±1 or ±2 in the
   option order, so a tile size moves to a nearby one), otherwise to any
   option. The cost model scores every new candidate.
3. Measure the best-scored unmeasured candidates seen during the
   evolution, plus random unmeasured ones for a fifth of the batch (3 of
   16). The random share keeps the
   model learning about regions it currently scores low.
4. Retrain the cost model on every measurement so far.

Until the model has 8 measurements with different times it cannot rank
anything, and the search proposes random points.

## The cost model

### Features

The model never sees configurations, only features of the scheduled
`PrimFunc` (`tessera/tune/features.py`), so one model serves every task
and every template. Each statement (a `BufferStore`, or an `Evaluate` of a
tensorized intrinsic) is described by 98 numbers:

- the six innermost loops on its path: extent and kind (serial, parallel,
  vectorized, unrolled, thread-bound);
- iterations, floating-point operations per iteration and in total,
  vector width, unroll product, parallel extent and depth, threadgroups
  and threads per group, whether a predicate guards it (imperfect splits),
  whether it is an intrinsic call or a reduction's initialization;
- for the written buffer and the two largest read buffers: the stride of
  the innermost loop, whether it is unit stride, reuse (accesses per
  distinct element), and the bytes touched while the innermost 1, 2, ...,
  8 loops run (a footprint curve, which is where cache capacity shows up);
- arithmetic intensity (operations per byte touched) at the same 8 levels.

Bytes touched come from the index expressions: an affine index spans
`1 + Σ |coefficient| × (extent − 1)` over the loops that vary, other
indices (`floordiv`/`floormod` from fused loops) are bounded by interval
arithmetic, and each span is clipped to the buffer's shape. A function's
vector is the vector of its heaviest statement, the elementwise maximum
over its other statements and six function totals (statements, blocks,
threadgroup and local bytes, total operations and iterations): 202 numbers.
Sizes are stored as `log2(1 + x)`.

### The model

The model (`tessera/tune/gbdt.py`, `tessera/tune/model.py`) is
gradient-boosted regression trees written in NumPy: 120 trees of depth up
to 6, learning rate 0.12, 80% row subsampling, features cut into 32
quantile bins, splits found from one histogram per node (the larger
child's histogram is the parent's minus the smaller child's). The target
is each measurement's throughput relative to the best measurement of its
task, `best / seconds`, and 0 for failures. Times of different tasks are
not comparable, but these ratios are, and the scale spends the model's
capacity on separating good configurations from each other rather than on
how slow the bad ones are. Retraining from scratch is cheap (about a
second for 2,000 rows of 200 features, much less early in a search), so
the model is refitted after every round.

Trees were chosen over a small neural network because the features are on
very different scales, the cost surface has thresholds (a tile that no
longer fits in L1, a register tile that spills), and only rankings within
a task matter; a network would need feature normalization and more data
before its rankings became useful.

## Spending the budget across tasks

`tune` (`tessera/tune/tuner.py`) works on all of a model's tasks with one
budget. Each round of 16 measurements goes to the task with the largest
expected saving of model time, `weight × best_seconds × rate`, where
`rate` is the task's relative improvement per round over its last two
rounds (0.5 until it has had two rounds, at least 0.005). Tasks that have
not been measured go first; tasks whose space is exhausted, or where
nothing has run successfully, go last. The budget includes what the log
already holds, which is what makes a rerun resume instead of start over.

## Testing

The strategies are tested on synthetic tasks (`tessera/tune/synthetic.py`):
a tiled matrix multiplication whose configuration chooses three-level
tiles for M and N, a split of K, where the outer K loop goes, unrolling
and vectorization. `schedule` builds the loop nest a CPU template would
produce, so the real feature extractor runs on it, and `measure` returns
the time of a simple machine model (FMA and load throughput, register
spills, L1 and L2 footprints, threads fed by the outer loop) with a
deterministic ±1.5% perturbation. The costs are a pure function of the
configuration, so a search with a fixed seed is reproducible and the
optimum of a 44,000-point space is found exhaustively in a fraction of a
second. The tests check that model-guided search finds more than 90% of
the optimal throughput within 128 trials and beats random search by at
least 5 points (`tests/tune/test_search.py`), that the model's top
candidates are mostly in the fastest tenth of unseen points, and that the
runner survives crashes, hangs and unimportable tasks
(`tests/tune/test_measure.py`). `MeasuredMatmul` runs the same loop nests
through the C backend, and `tests/tune/test_measured.py` tunes one end to
end with real measurements.

## Limitations

- The features describe loops and memory, not instructions. Two schedules
  that lower to the same loop nest but different microkernels look the
  same unless the intrinsic call differs.
- Footprints are upper bounds for strided and non-affine accesses, and
  they ignore cache-line granularity, so the model learns cache effects
  only as far as tile sizes reveal them.
- The model is retrained from scratch each round and does not carry
  knowledge between tuning runs except through the log.
- Measurements are wall-clock medians on a shared machine; the tuner does
  not detect interference, so logs from a busy machine favour whatever ran
  in quiet moments.
