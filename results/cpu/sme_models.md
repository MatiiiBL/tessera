# SME threads on whole models

Whole models compiled for c:sme (fusion, the templates' defaults, one library): as compiled, and with the microkernels' thread count forced to one and to every performance core. Each variant's best median of 10 runs over interleaved rounds.

Measured 2026-11-12T11:59:36+00:00 on Apple M4 (4 performance cores), Darwin 27.0.0, Apple clang version 21.0.0 (clang-2100.0.123.102), NumPy 2.5.4; load average 1.32 at the start. Latency in ms (lower is better).

| case | default | SME on 1 thread | SME on every thread |
|---|---:|---:|---:|
| resnet50 | 15.6 | 23.7 | 16.9 |
| bert | 22.0 | 25.7 | 21.1 |
| gpt2 | 32.4 | 37.9 | 32.6 |

Reproduce: `uv run python -m tessera.backend.c.bench`.
