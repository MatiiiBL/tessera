# SME and threads

The SME GEMM microkernel called from one to four threads, each on its own band of rows of C, against the NEON microkernel on the same bands; then one thread running SME on the given share of the rows while three run NEON on the rest.

Measured 2026-11-12T07:24:48+00:00 on Apple M4 (4 performance cores), Darwin 27.0.0, Apple clang version 21.0.0 (clang-2100.0.123.102), NumPy 2.5.4; load average 3.45 at the start. Throughput in GFLOPS (higher is better).

| case | sme t1 | neon t1 | sme t2 | neon t2 | sme t3 | neon t3 | sme t4 | neon t4 | sme t1 + neon t3, 70% of rows on sme | sme t1 + neon t3, 75% of rows on sme | sme t1 + neon t3, 80% of rows on sme | sme t1 + neon t3, 85% of rows on sme | sme t1 + neon t3, 90% of rows on sme | sme t1 + neon t3, 95% of rows on sme |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024x1024x1024 | 1342 | 122 | 1585 | 239 | 1563 | 346 | 1624 | 406 | 869 | 1088 | 1161 | 1341 | 1357 | 1271 |
| 128x3072x768 | 1103 | 116 | 1071 | 219 | 897 | 157 | 595 | 274 | 432 | 550 | 542 | 518 | 522 | 516 |

Reproduce: `uv run python -m tessera.backend.c.bench`.
