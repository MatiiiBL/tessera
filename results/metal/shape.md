# Metal: threadgroup shape

Measured 2026-11-12T07:35:32Z on Apple M4 (4 performance cores, 10 cores, 16 GB), Darwin 27.0.0, commit `fac1e9b`. Reproduce with `uv run python -m tessera.backend.metal.bench shape`.

One SIMD-group matrix multiply kernel (each SIMD group holds 2 x 4 accumulator tiles of 8 x 8,
loads A and B from device memory), generated twice: with the threadgroup shaped like its thread
loops (32 lanes x SIMD-group columns x SIMD-group rows), and launched flat with the positions
computed from the thread index. GFLOPS from GPU time.

| Size | SIMD groups | Shape | Shaped (GFLOPS) | Flat (GFLOPS) | Speedup |
|---|---|---|---|---|---|
| 1024³ | 1 x 2 | 32 x 2 x 1 | 2586 | 2603 | 1.01 |
| 1024³ | 2 x 1 | 32 x 1 x 2 | 2573 | 2572 | 1.00 |
| 1024³ | 2 x 2 | 32 x 2 x 2 | 2531 | 2558 | 1.01 |
| 2048³ | 1 x 2 | 32 x 2 x 1 | 1779 | 2565 | 1.44 |
| 2048³ | 2 x 1 | 32 x 1 x 2 | 2514 | 2535 | 1.01 |
| 2048³ | 2 x 2 | 32 x 2 x 2 | 1591 | 2550 | 1.60 |
