# SME and threads

The SME GEMM microkernel called from one to four threads, each on its own band of rows of C, against the NEON microkernel on the same bands; then one thread running SME on the given share of the rows while three run NEON on the rest.

Measured 2026-11-12T11:59:28+00:00 on Apple M4 (4 performance cores), Darwin 27.0.0, Apple clang version 21.0.0 (clang-2100.0.123.102), NumPy 2.5.4; load average 1.35 at the start. Throughput in GFLOPS (higher is better).

| case | sme t1 | neon t1 | sme t2 | neon t2 | sme t3 | neon t3 | sme t4 | neon t4 | sme t1 + neon t3, 70% of rows on sme | sme t1 + neon t3, 75% of rows on sme | sme t1 + neon t3, 80% of rows on sme | sme t1 + neon t3, 85% of rows on sme | sme t1 + neon t3, 90% of rows on sme | sme t1 + neon t3, 95% of rows on sme |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024x1024x1024 | 1116 | 127 | 1460 | 237 | 1536 | 347 | 1701 | 239 | 881 | 1069 | 1237 | 1385 | 1389 | 1272 |
| 128x3072x768 | 1049 | 115 | 1363 | 217 | 995 | 268 | 882 | 201 | 370 | 432 | 433 | 431 | 432 | 432 |

Reproduce: `uv run python -m tessera.backend.c.bench`.
