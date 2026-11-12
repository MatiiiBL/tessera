# SME tiles and threads

The matmul template's SME path (tiles of C computed by the microkernel, with the init and the copy of each tile) with its tiles on one, two or four threads, the tiles sized for each count as the defaults would size them. Each variant's best time over interleaved rounds.

Measured 2026-11-12T07:01:22+00:00 on Apple M4 (4 performance cores), Darwin 27.0.0, Apple clang version 21.0.0 (clang-2100.0.123.102), NumPy 2.5.4; load average 7.99 at the start. Throughput in GFLOPS (higher is better).

| case | t1 | t2 | t4 |
|---|---:|---:|---:|
| 128x768x768 | 446 | 627 | 759 |
| 128x768x3072 | 440 | 671 | 879 |
| 128x3072x768 | 471 | 677 | 892 |

Reproduce: `uv run python -m tessera.backend.c.bench`.
