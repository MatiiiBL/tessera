# Metal: threadgroup shape

**Provisional:** measured while the machine was busy (load average 7.07 at the start). Re-run on a quiet machine for final numbers.

Measured 2026-11-12T05:57:00Z on Apple M4 (4 performance cores, 10 cores, 16 GB), Darwin 27.0.0, commit `e854764`. Reproduce with `uv run python -m tessera.backend.metal.bench shape`.

One SIMD-group matrix multiply kernel (each SIMD group holds 2 x 4 accumulator tiles of 8 x 8,
loads A and B from device memory), generated twice: with the threadgroup shaped like its thread
loops (32 lanes x SIMD-group columns x SIMD-group rows), and launched flat with the positions
computed from the thread index. GFLOPS from GPU time.

| Size | SIMD groups | Shape | Shaped (GFLOPS) | Flat (GFLOPS) | Speedup |
|---|---|---|---|---|---|
| 1024³ | 1 x 2 | 32 x 2 x 1 | 1899 | 1970 | 1.04 |
| 1024³ | 2 x 1 | 32 x 1 x 2 | 1397 | 1289 | 0.92 |
| 1024³ | 2 x 2 | 32 x 2 x 2 | 1610 | 1471 | 0.91 |
| 2048³ | 1 x 2 | 32 x 2 x 1 | 1533 | 1550 | 1.01 |
| 2048³ | 2 x 1 | 32 x 1 x 2 | 1383 | 1388 | 1.00 |
| 2048³ | 2 x 2 | 32 x 2 x 2 | 1448 | 1399 | 0.97 |
