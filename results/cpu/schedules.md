# Schedule choices

Matmul rows: the matmul template's generated loops (c:generic, one thread) with the column strips of B outside the row tiles of A (the default, B packed into a contiguous local copy per strip when it fits the 64 KiB stack limit), without the packing, and with the row tiles outside. Convolution rows: a ResNet-50 layer on NEON with every thread, computed by the implicit-GEMM microkernel or by the generated loops. Each variant's best time over interleaved rounds.

Measured 2026-11-12T07:24:40+00:00 on Apple M4 (4 performance cores), Darwin 27.0.0, Apple clang version 21.0.0 (clang-2100.0.123.102), NumPy 2.5.4; load average 3.52 at the start. Throughput in GFLOPS (higher is better).

| case | column strips outside, B packed | column strips outside | row tiles outside | conv: microkernel | conv: generated loops |
|---|---:|---:|---:|---:|---:|
| 128x768x768 | 114 | 107 | 110 |  |  |
| 128x768x3072 | 103 | 103 | 105 |  |  |
| 128x3072x768 | 113 | 91 | 95 |  |  |
| layer1 3x3, neon t4 |  |  |  | 238 | 369 |
| layer3 3x3, neon t4 |  |  |  | 353 | 225 |
| layer4 1x1 reduce, neon t4 |  |  |  | 397 | 229 |

Reproduce: `uv run python -m tessera.backend.c.bench`.
