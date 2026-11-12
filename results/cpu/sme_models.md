# SME threads on whole models

Whole models compiled for c:sme (fusion, the templates' defaults, one library): as compiled, and with the microkernels' thread count forced to one and to every performance core. Each variant's best median of 10 runs over interleaved rounds.

Measured 2026-11-12T07:31:18+00:00 on Apple M4 (4 performance cores), Darwin 27.0.0, Apple clang version 21.0.0 (clang-2100.0.123.102), NumPy 2.5.4; load average 3.08 at the start. Latency in ms (lower is better).

| case | default | SME on 1 thread | SME on every thread |
|---|---:|---:|---:|
| resnet50 | 18.2 | 23.9 | 17.8 |
| bert | 27.1 | 31.0 | 26.1 |
| gpt2 | 38.0 | 44.3 | 55.9 |

Reproduce: `uv run python -m tessera.backend.c.bench`.
