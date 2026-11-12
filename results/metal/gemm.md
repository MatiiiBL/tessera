# Metal: GEMM throughput (GFLOPS)

**Provisional:** measured while the machine was busy (load average 6.24 at the start). Re-run on a quiet machine for final numbers.

Measured 2026-11-12T06:54:03Z on Apple M4 (4 performance cores, 10 cores, 16 GB), Darwin 27.0.0, commit `cb94edd`. Reproduce with `uv run --group bench python -m tessera.backend.metal.bench gemm`.

float32 C = A @ B, B a constant. GPU time per call: Metal timestamps around batches of launches
for Tessera, MPS events around 50 calls for PyTorch 2.15.0. GFLOPS count 2 M N K per batch.

| Shape (batch, M, N, K) | Tessera default | MPS |
|---|---|---|
| 256 x 256 x 256 | 1827 | 276 |
| 512 x 512 x 512 | 2301 | 1654 |
| 1024 x 1024 x 1024 | 2310 | 2163 |
| 2048 x 2048 x 2048 | 2104 | 1578 |
| 4096 x 4096 x 4096 | 1056 | 1614 |
| attention projection (BERT, GPT-2) | 1240 | 985 |
| fused QKV projection (GPT-2) | 1511 | 1324 |
| MLP up (BERT, GPT-2) | 1912 | 1534 |
| MLP down (BERT, GPT-2) | 1739 | 1307 |
| attention scores, 12 heads | 1534 | 276 |
| attention times values, 12 heads | 1711 | 322 |
