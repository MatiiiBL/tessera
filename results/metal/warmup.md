# Metal: GPU clock warmup

Measured 2026-11-12T07:35:26Z on Apple M4 (4 performance cores, 10 cores, 16 GB), Darwin 27.0.0, commit `fac1e9b`. Reproduce with `uv run python -m tessera.backend.metal.bench warmup`.

A compute-bound kernel of 256 x 256 threads, in batches of 10 (one command buffer each,
the CPU waiting for each), after the GPU has been idle for 3 s. The last column repeats
the batches after one continuous command buffer of 34 ms of the same kernel.

| Batch | GPU time so far (ms) | Per launch, from idle (µs) | Per launch, after warmup (µs) |
|---|---|---|---|
| 1 | 13.3 | 1332.6 | 609.9 |
| 2 | 19.5 | 617.0 | 616.9 |
| 3 | 25.7 | 620.7 | 607.9 |
| 5 | 38.0 | 613.4 | 595.0 |
| 10 | 67.0 | 546.7 | 541.4 |
| 20 | 119.1 | 498.3 | 489.9 |
| 40 | 212.0 | 434.8 | 436.4 |
