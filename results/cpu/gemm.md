# GEMM throughput

fp32 C = A B with A, B and C row-major. `generic`, `neon` and `sme` are the kernels the matmul template generates for those targets; `numpy` is `numpy.matmul`, which calls Accelerate's sgemm on macOS. `t1` is one thread, `t4` every performance core.

Measured 2026-11-12T07:28:53+00:00 on Apple M4 (4 performance cores), Darwin 27.0.0, Apple clang version 21.0.0 (clang-2100.0.123.102), NumPy 2.5.4; load average 1.77 at the start. Throughput in GFLOPS (higher is better).

| case | numpy t1 | numpy t4 | generic t1 | generic t4 | neon t1 | neon t4 | sme t1 | sme t4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 256x256x256 | 1412 | 1417 | 117 | 374 | 118 | 356 | 1112 | 1114 |
| 512x512x512 | 1767 | 1764 | 116 | 379 | 122 | 407 | 1474 | 630 |
| 1024x1024x1024 | 1556 | 1852 | 114 | 360 | 123 | 412 | 1444 | 999 |
| 2048x2048x2048 | 1640 | 1861 | 63 | 186 | 120 | 400 | 1354 | 1355 |
| 4096x4096x4096 | 1584 | 1824 | 23 | 43 | 118 | 374 | 1245 | 1291 |
| 128x768x768 | 1686 | 1687 | 108 | 331 | 119 | 380 | 1496 | 1520 |
| 128x768x3072 | 1624 | 1499 | 94 | 290 | 120 | 383 | 1361 | 1417 |
| 128x3072x768 | 1539 | 1276 | 107 | 336 | 116 | 386 | 1335 | 752 |

Reproduce: `uv run python -m tessera.backend.c.bench`.
