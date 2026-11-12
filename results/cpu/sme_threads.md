# SME and threads

The SME GEMM microkernel called from one to four threads, each on its own band of rows of C, against the NEON microkernel on the same bands; then one thread running SME on the given share of the rows while three run NEON on the rest.

Measured 2026-11-12T06:55:17+00:00 on Apple M4 (4 performance cores), Darwin 27.0.0, Apple clang version 21.0.0 (clang-2100.0.123.102), NumPy 2.5.4; load average 6.33 at the start. Throughput in GFLOPS (higher is better).

| case | sme t1 | neon t1 | sme t2 | neon t2 | sme t3 | neon t3 | sme t4 | neon t4 | sme t1 + neon t3, 70% of rows on sme | sme t1 + neon t3, 75% of rows on sme | sme t1 + neon t3, 80% of rows on sme | sme t1 + neon t3, 85% of rows on sme | sme t1 + neon t3, 90% of rows on sme | sme t1 + neon t3, 95% of rows on sme |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024x1024x1024 | 1324 | 120 | 1444 | 234 | 1338 | 316 | 1338 | 378 | 769 | 890 | 886 | 915 | 887 | 832 |
| 128x3072x768 | 572 | 76 | 712 | 140 | 586 | 123 | 818 | 227 | 290 | 347 | 355 | 348 | 352 | 347 |

Reproduce: `uv run python -m tessera.backend.c.bench`.
