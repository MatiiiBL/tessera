# Metal: GEMM throughput (GFLOPS)

Measured 2026-11-12T07:35:41Z on Apple M4 (4 performance cores, 10 cores, 16 GB), Darwin 27.0.0, commit `fac1e9b`. Reproduce with `uv run --group bench python -m tessera.backend.metal.bench gemm --trials 16`.

float32 C = A @ B, B a constant. GPU time per call: Metal timestamps around batches of launches
for Tessera, MPS events around 50 calls for PyTorch 2.15.0. GFLOPS count 2 M N K per batch.

| Shape (batch, M, N, K) | Tessera default | Tessera best of 16 | MPS |
|---|---|---|---|
| 256 x 256 x 256 | 1796 | 1894 | 369 |
| 512 x 512 x 512 | 2634 | 2634 | 2070 |
| 1024 x 1024 x 1024 | 2691 | 2691 | 2560 |
| 2048 x 2048 x 2048 | 2620 | 2620 | 2750 |
| 4096 x 4096 x 4096 | 1253 | 1270 | 1828 |
| attention projection (BERT, GPT-2) | 1640 | 1973 | 1597 |
| fused QKV projection (GPT-2) | 1749 | 1869 | 2255 |
| MLP up (BERT, GPT-2) | 2449 | 2449 | 2114 |
| MLP down (BERT, GPT-2) | 1972 | 1972 | 1922 |
| attention scores, 12 heads | 1484 | 1495 | 697 |
| attention times values, 12 heads | 1560 | 1657 | 735 |
