# Schedule choices

Matmul rows: the matmul template's generated loops (c:generic, one thread) with the column strips of B outside the row tiles of A (the default, B packed into a contiguous local copy per strip when it fits the 64 KiB stack limit), without the packing, and with the row tiles outside. Convolution rows: a ResNet-50 layer on NEON with every thread, computed by the implicit-GEMM microkernel or by the generated loops. Each variant's best time over interleaved rounds.

Measured 2026-11-12T07:02:15+00:00 on Apple M4 (4 performance cores), Darwin 27.0.0, Apple clang version 21.0.0 (clang-2100.0.123.102), NumPy 2.5.4; load average 4.85 at the start. Throughput in GFLOPS (higher is better).

| case | column strips outside, B packed | column strips outside | row tiles outside | conv: microkernel | conv: generated loops |
|---|---:|---:|---:|---:|---:|
| 128x768x768 | 112 | 106 | 109 |  |  |
| 128x768x3072 | 103 | 101 | 103 |  |  |
| 128x3072x768 | 113 | 91 | 94 |  |  |
| layer1 3x3, neon t4 |  |  |  | 240 | 367 |
| layer3 3x3, neon t4 |  |  |  | 230 | 168 |
| layer4 1x1 reduce, neon t4 |  |  |  | 154 | 149 |

Reproduce: `uv run python -m tessera.backend.c.bench`.
