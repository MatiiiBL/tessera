# Metal: GPU clock warmup

**Provisional:** measured while the machine was busy (load average 6.28 at the start). Re-run on a quiet machine for final numbers.

Measured 2026-11-12T05:54:25Z on Apple M4 (4 performance cores, 10 cores, 16 GB), Darwin 27.0.0, commit `e854764`. Reproduce with `uv run python -m tessera.backend.metal.bench warmup`.

A compute-bound kernel of 256 x 256 threads, in batches of 10 (one command buffer each,
the CPU waiting for each), after the GPU has been idle for 3 s. The last column repeats
the batches after one continuous command buffer of 24 ms of the same kernel.

| Batch | GPU time so far (ms) | Per launch, from idle (µs) | Per launch, after warmup (µs) |
|---|---|---|---|
| 1 | 15.9 | 1592.2 | 413.4 |
| 2 | 22.1 | 615.7 | 389.5 |
| 3 | 28.2 | 611.5 | 388.1 |
| 5 | 36.9 | 388.5 | 391.8 |
| 10 | 56.3 | 388.2 | 388.9 |
| 20 | 95.2 | 389.0 | 386.1 |
| 40 | 172.8 | 393.2 | 388.5 |
