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
groups with equal keys and counts them in `weight`. The key covers the
loop nest but not which inputs are constants, and the backends treat
constants differently (the C backend packs weights once, at load), so two
groups with the same loop nest whose constant inputs differ, such as a
matmul by a weight and a matmul of two activations of the same shapes,
are refused as one task with an error instead of being tuned for each
other; none of the benchmark models has such a pair. The target is part of
each log record (`Target.key` includes the ISA, the thread count and the
CPU name), so a log tuned on one machine is never applied to another.

## Measuring candidates

`LocalRunner` (`tessera/tune/measure.py`) handles a batch of candidates in
two phases:

1. **Build** in parallel processes (`build_workers`, 3 by default). A
   build applies the template to the configuration and, when the backend
   is a `Prebuilder` (both backends are), compiles the kernel into the
   build cache. Each build runs a compiler (clang, or Metal's) with its
   own CPU and memory, so the count bounds what a batch takes from the
   machine. Builds
   and measurements never overlap, so the count does not affect timings.
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
workers now start their own interpreter. A job's timeout starts only once
its process reports that it has started: importing NumPy and the compiler
can take seconds on a busy machine, and counting that against the first
job turned healthy candidates into timeouts.

The build processes also score candidates for the search. Scoring a
candidate means applying the template and extracting the features of the
result, which takes 3 to 9 ms in Python (Metal's templates are the slow
ones), and an evolutionary round scores a few hundred new candidates. In
the first 28-minute Metal run, scoring in the tuner's own process, 70% of
the time went to the search rather than to building and measuring, and it
made 2,546 measurements; with scoring sent in batches of 32 to the three
build processes, which are idle while the search runs, the next two runs
of the same length made 4,000 and 3,840.

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
current template still accepts (see Validation below for how it chooses),
so a log written before a template gained or renamed a knob falls back to
the next valid record or to the defaults.

Task keys changed once, when they started to cover which inputs are
constants (see Tasks). The Metal log was migrated then, with a one-time
script: 11,917 of its 14,294 records were renamed to the new keys of the
same functions, 137 kept keys that did not change, and the 2,240 records
of the 8 matmul tasks whose functions a fix in the Metal backend had
changed were dropped; those tasks were tuned again. The `c:sme` log was
discarded instead, because the C backend's timing harness changed (it now
prepares constant inputs the way a loaded model does).

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
4. Retrain the cost model on every measurement so far, once they have
   grown by 10% since the last fit (or by 16, early on): with thousands of
   measurements one round adds little, and a fit takes about a second.

Until the model has 8 measurements with different times it cannot rank
anything, and the search proposes random points.

**Validation.** The search measures each candidate once, and measurements
taken far apart in time are not comparable: noise only ever slows a
measurement, and a machine that warms up during a long run slows
everything. In the first `c:sme` run, one task's default measured 744 µs
in its first round and 1,067 to 1,103 µs at the end, so the fastest record
overall compared configurations measured at different temperatures. After
the search, `tessera tune` therefore runs two validation passes, and
`compile` decides each task by its latest pass alone; its records carry
the pass's name in an optional `validation` field (`BestConfigs`, with the
plain minimum over all records for a task that was never validated).

1. *Measured alone:* each task's three fastest configurations and its
   default, three times, interleaved round-robin across tasks. Within a
   pass the contenders are measured side by side, so the minimum over a
   pass compares like with like, and a configuration that is really slower
   than the default loses to it (`tests/tune/test_tuner.py`).
2. *In the model* (`tessera/tune/inmodel.py`): each model is compiled with
   the logs and with the defaults, the two are profiled in alternation, and
   for every task whose tuned configuration differs from its default the
   two in-model times per kernel are logged. This pass exists because a
   kernel alone is not always the kernel in its model. In the timing
   harness a kernel's weight is one buffer, used call after call and
   received like any other input; in a model each kernel has its own
   weight, prepared once when the model is loaded, and twelve layers'
   weights stream through the caches in turn. In GPT-2, a matmul shared by
   twelve kernels took 1,065 µs alone with its default configuration (which
   packs the weight) and 640 µs with the tuned one (which does not), while
   in the model the default took 0.72 ms and the tuned configuration
   1.12 ms. Before this pass existed, the tuned GPT-2 took 56.6 ms on
   `c:sme` against 47.4 ms untuned. The in-model pass keeps the default for
   every such task, whatever the harness gets wrong; the C backend's
   `measure` is also being changed to prepare constant inputs the way a
   loaded model does.

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

`tune` (`tessera/tune/tuner.py`) works on all the tasks of several models
with one budget; tasks the models share are tuned once, weighted by the
number of kernels that use them. Each round of 16 measurements goes to the
task with the largest expected saving of model time,
`weight × best_seconds × rate`, where `rate` is the task's expected
relative improvement per round: the larger of what its last two rounds
achieved and an optimistic `0.5 / rounds`, with `rounds` counted from the
measurements the task already has (log included), and at least 0.005.
Tasks that have not been measured go first; tasks whose space is
exhausted, or where nothing has run successfully, go last. The budget
includes what the log already holds, which is what makes a rerun resume
instead of start over.

The optimistic term came from a flaw the first full tuning run exposed.
The scheduler then used only the last two rounds' improvement (0.5 before
a task had two rounds), so a task whose second round found nothing fell
to the 0.005 floor at once, while the slow 128×50257×768 matmul of GPT-2's
vocabulary projection, the largest single kernel, kept improving a little
each round and took the budget. Of that run's 4,476 measurements on
`c:sme`, this one task got 1,404 (31%); the median task got 41 (12 of them
from validation) and none fewer than 21, so most tasks shared by twelve
kernels had two rounds. Two rounds without progress say little about a
space of thousands of configurations. With `0.5 / rounds`, a task stays in
the running for a number of rounds roughly proportional to its share of
the model's time. Resuming `c:sme` with the fix, 7,349 more measurements
went first to the matmuls that BERT's and GPT-2's twelve layers share (the
six busiest tasks got 1,132, 1,116, 700, 684, 588 and 508), and the busiest
task's share fell to 15%; the Metal run, made with the fix from the start,
gave its busiest task 9.2% of 14,294 measurements and the median task 44.

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

## Tuning the four models

The committed logs (`results/tuning/c-sme.jsonl` and `metal.jsonl`) were
made with the final templates of both backends on a machine with nothing
else running, with one command per target:

```sh
uv run tessera tune data/models/{resnet50,mobilenet_v2,bert,gpt2}.onnx -t c:sme --time-budget 2700
# Metal in runs of at most 28 minutes; each run resumes the log
uv run tessera tune data/models/{resnet50,mobilenet_v2,bert,gpt2}.onnx -t metal --time-budget 1680
```

The four models have 78 tasks on each target (355 kernels); the spaces of
the final templates hold 3.1 million configurations on `c:sme` and 420
million on Metal, so the logs come from search, not enumeration.

| | `c:sme` | Metal |
|---|---|---|
| machine time | 47 min | 141 min |
| measurements | 9,321 | 16,330 |
| failed | 1 (a 60 s timeout) | 340 (299 rejected by a template, 41 build errors and killed processes) |
| busiest task's share of the search | 13.7% | 8.5% |
| tasks where the in-model pass kept the default | 16 of 67, then 4 of 51 more | 16 of 67 |

`c:sme` was tuned from scratch in one 46-minute run (45 minutes of search,
then both validation passes), after the C backend's timing harness changed
to prepare constant inputs the way a loaded model does; its in-model pass
was run once more, a minute long, after a runtime fix, and kept the default
for 4 more of the 51 tasks still using a tuned configuration. Metal's 141
minutes, more than the two hours planned, are three runs: 114 minutes in
four chunks over all tasks, 20 minutes for the 8 matmul tasks a fix in the
Metal backend changed (`--new-only`, after the key migration described in
Tuning logs), and 7 minutes validating all 78 tasks alone and in the
models. On both targets the in-model pass found about a quarter of the
tuned configurations slower inside the model than the defaults; in the
first tuning run, before this pass existed, the tuned GPT-2 was slower on
`c:sme` than the untuned one.

Tuned against untuned, the median model latency
([results/models.md](../results/models.md), CPU runs pooled from fresh
processes):

| model | c:sme, ms | Metal, ms |
|---|---|---|
| ResNet-50 | 12.02 to 11.52 (1.04 times) | 16.92 to 10.71 (1.58 times) |
| MobileNetV2 | 3.15 to 2.15 (1.47 times) | 1.95 to 1.74 (1.12 times) |
| BERT-base | 21.98 to 18.06 (1.22 times) | 10.54 to 10.31 (1.02 times) |
| GPT-2 | 32.36 to 32.68 (0.99 times) | 14.65 to 14.18 (1.03 times) |

## Results: random against model-guided search

`tessera bench tuning` runs each strategy from scratch with several seeds
and reports, after each number of measurements, the best throughput found
so far relative to a reference ([results/tuning.md](../results/tuning.md),
measured 2026-11-12 with nothing else running). Both strategies start with
the same round (the default and 15 random points: the model has nothing to
learn from yet), so they differ from the second round on.

**Synthetic tasks** (the reference is the exhaustive optimum; the costs are
analytic, so the results do not depend on the machine), mean over 5 seeds:

| task (M×N×K, space size) | trials | random | model-guided |
|---|---|---|---|
| 64×64×64 (43,904) | 64 / 256 | 0.73 / 0.83 | 0.92 / 0.99 |
| 256×128×192 (181,440) | 64 / 256 | 0.81 / 0.88 | 0.88 / 0.98 |
| 32×512×64 (64,680) | 64 / 256 | 0.72 / 0.84 | 0.88 / 0.97 |

**Real tasks:** the two heaviest tasks of ResNet-50 and of BERT on `c:sme`,
3 seeds of 96 measurements each. A task whose space has at most 2,048
configurations is measured exhaustively once and every search replays
those measurements, which makes the reference the true optimum and gives
every strategy the same noise. The final templates' spaces are far larger
(6,480 to 368,640 configurations), so here every search measured its own
candidates, and the reference is the best time any run found:

| task (groups sharing it, space size) | trials | random | model-guided |
|---|---|---|---|
| ResNet-50 3×3 conv, 56×56×64 → 64 (3, 8,400) | 32 / 96 | 0.92 / 0.98 | 0.92 / 0.92 |
| ResNet-50 3×3 conv, 14×14×256 → 256 (5, 6,480) | 32 / 96 | 0.93 / 0.97 | 0.99 / 0.99 |
| BERT matmul 128×768×3072, M×N×K (12, 350,208) | 32 / 96 | 0.69 / 0.79 | 0.75 / 0.84 |
| BERT matmul 128×3072×768 (12, 368,640) | 32 / 96 | 0.87 / 0.88 | 0.96 / 0.98 |

These live measurements are noisy on `c:sme`: the thread placement of each
measurement process varies (docs/cpu.md, Limitations), and the two
strategies' curves already differ by up to 16% at 8 trials, where both have
measured the same configurations. With that in mind, the model-guided
search ends clearly ahead on one task (the second matmul: 0.98 against
0.88), ahead within the noise on two, and behind on one convolution, where
it stopped improving after 32 trials while random search kept finding
better points. The synthetic comparison, which has no measurement noise,
shows the difference the method makes; on the real tasks, with 96
measurements per search and this much noise, it is smaller.

## Limitations

- The features describe loops and memory, not instructions. Two schedules
  that lower to the same loop nest but different microkernels look the
  same unless the intrinsic call differs.
- Footprints are upper bounds for strided and non-affine accesses, and
  they ignore cache-line granularity, so the model learns cache effects
  only as far as tile sizes reveal them.
- The model is retrained from scratch when it refits and carries no
  knowledge between tuning runs except through the log.
- Measurements are wall-clock medians; the tuner does not detect
  interference. The committed logs were measured on a machine with nothing
  else running, and the validation pass guards the final choice, but a log
  measured on a busy machine favours whatever ran in quiet moments.
- The budget covers a tiny part of the spaces of the final templates
  (3.1 million configurations on `c:sme` and 420 million on Metal, against
  11,825 and 14,294 measurements), so the logs hold good configurations,
  not optimal ones.
- Even with scoring in the build processes, about half of the Metal
  tuning time went to the search rather than to measurements.
- Metal's templates reject some points of their own spaces when a schedule
  primitive cannot apply (2.4% of the candidates tried); these are recorded
  as failed builds and cost a template application each.
