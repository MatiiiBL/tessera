# GEMM throughput

fp32 C = A B with A, B and C row-major. `generic`, `neon` and `sme` are the kernels the matmul template generates for those targets; `numpy` is `numpy.matmul`, which calls Accelerate's sgemm on macOS. `t1` is one thread, `t4` every performance core.

Measured 2026-11-12T12:02:16+00:00 on Apple M4 (4 performance cores), Darwin 27.0.0, Apple clang version 21.0.0 (clang-2100.0.123.102), NumPy 2.5.4; load average 1.72 at the start. Throughput in GFLOPS (higher is better).

| case | numpy t1 | numpy t4 | generic t1 | generic t4 | neon t1 | neon t4 | sme t1 | sme t4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 256x256x256 | 1419 | 1407 | 115 | 382 | 116 | 363 | 1110 | 1112 |
| 512x512x512 | 1764 | 1760 | 116 | 384 | 122 | 418 | 1485 | 1560 |
| 1024x1024x1024 | 1687 | 1891 | 116 | 381 | 123 | 437 | 1532 | 1633 |
| 2048x2048x2048 | 1707 | 1919 | 64 | 206 | 121 | 415 | 1372 | 1362 |
| 4096x4096x4096 | 1600 | 1874 | 23 | 65 | 120 | 394 | 1298 | 1292 |
| 128x768x768 | 1740 | 1737 | 113 | 346 | 123 | 403 | 1581 | 1582 |
| 128x768x3072 | 1743 | 1550 | 101 | 309 | 119 | 404 | 1450 | 1547 |
| 128x3072x768 | 1601 | 1282 | 111 | 349 | 122 | 408 | 1385 | 1496 |

Reproduce: `uv run python -m tessera.backend.c.bench`.
