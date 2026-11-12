# Loop-IR interpreter speed

Measured 2026-11-12T06:08:19Z on arm64, Python 3.13.7, 1-minute load average 6.8 (the machine was shared; numbers are provisional).
Regenerate with `uv run python -m tessera.tir.bench`.

| kernel | block executions | compile (ms) | run (ms) | ns per execution |
|---|---:|---:|---:|---:|
| matmul 64x64x64 | 262,144 | 0.6 | 81.8 | 312 |
| matmul 64x64x64, tiled 8x8 | 266,240 | 0.8 | 99.1 | 372 |
| conv2d 3x3 NHWC 16x16x16 -> 16, relu | 593,920 | 1.5 | 380.0 | 640 |
| softmax 64x512 | 98,304 | 0.8 | 22.3 | 227 |
| gelu 256x256 | 65,536 | 0.3 | 15.2 | 233 |

Cost of one read-modify-write `x[5] = x[5] + x[7] * x[8]` from Python:

| storage | ns |
|---|---:|
| array.array | 100 |
| list | 52 |
| numpy | 302 |
