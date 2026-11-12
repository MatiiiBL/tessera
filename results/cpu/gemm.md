# GEMM throughput

fp32 C = A B with A, B and C row-major. `generic`, `neon` and `sme` are the kernels the matmul template generates for those targets; `numpy` is `numpy.matmul`, which calls Accelerate's sgemm on macOS. `t1` is one thread, `t4` every performance core.

Measured 2026-11-12T06:49:22+00:00 on Apple M4 (4 performance cores), Darwin 27.0.0, Apple clang version 21.0.0 (clang-2100.0.123.102), NumPy 2.5.4; load average 4.32 at the start. Throughput in GFLOPS (higher is better).

| case | numpy t1 | numpy t4 | generic t1 | generic t4 | neon t1 | neon t4 | sme t1 | sme t4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 256x256x256 | 1231 | 1262 | 92 | 313 | 98 | 260 | 918 | 368 |
| 512x512x512 | 1454 | 1470 | 101 | 323 | 96 | 321 | 856 | 347 |
| 1024x1024x1024 | 1238 | 830 | 75 | 239 | 96 | 268 | 695 | 381 |
| 2048x2048x2048 | 1009 | 891 | 59 | 141 | 102 | 253 | 632 | 376 |
| 4096x4096x4096 | 1055 | 1185 | 18 | 28 | 74 | 194 | 491 | 434 |
| 128x768x768 | 1042 | 1070 | 74 | 192 | 75 | 209 | 614 | 331 |
| 128x768x3072 | 1082 | 975 | 70 | 155 | 79 | 226 | 506 | 346 |
| 128x3072x768 | 894 | 804 | 78 | 201 | 101 | 250 | 686 | 356 |

Reproduce: `uv run python -m tessera.backend.c.bench`.
