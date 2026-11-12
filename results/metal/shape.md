# Metal: threadgroup shape

**Provisional:** measured while the machine was busy (load average 6.22 at the start). Re-run on a quiet machine for final numbers.

Measured 2026-11-12T06:54:24Z on Apple M4 (4 performance cores, 10 cores, 16 GB), Darwin 27.0.0, commit `cb94edd`. Reproduce with `uv run python -m tessera.backend.metal.bench shape`.

One SIMD-group matrix multiply kernel (each SIMD group holds 2 x 4 accumulator tiles of 8 x 8,
loads A and B from device memory), generated twice: with the threadgroup shaped like its thread
loops (32 lanes x SIMD-group columns x SIMD-group rows), and launched flat with the positions
computed from the thread index. GFLOPS from GPU time.

| Size | SIMD groups | Shape | Shaped (GFLOPS) | Flat (GFLOPS) | Speedup |
|---|---|---|---|---|---|
| 1024³ | 1 x 2 | 32 x 2 x 1 | 1358 | 2325 | 1.71 |
| 1024³ | 2 x 1 | 32 x 1 x 2 | 1469 | 1378 | 0.94 |
| 1024³ | 2 x 2 | 32 x 2 x 2 | 1597 | 1572 | 0.98 |
| 2048³ | 1 x 2 | 32 x 2 x 1 | 1659 | 1664 | 1.00 |
| 2048³ | 2 x 1 | 32 x 1 x 2 | 1537 | 1588 | 1.03 |
| 2048³ | 2 x 2 | 32 x 2 x 2 | 1500 | 1586 | 1.06 |
