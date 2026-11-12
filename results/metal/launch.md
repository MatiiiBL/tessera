# Metal: the cost of a kernel launch

Measured 2026-11-12T07:35:25Z on Apple M4 (4 performance cores, 10 cores, 16 GB), Darwin 27.0.0, commit `fac1e9b`. Reproduce with `uv run python -m tessera.backend.metal.bench launch`.

A one-threadgroup kernel that updates one float, launched 1 to 1024 times in one
command buffer. Per-launch costs are the slope of the median time (over 30 runs) against the
number of launches; the last column is the round trip of a command buffer with one
launch, from encoding to the CPU seeing it complete.

| Encoding | GPU per launch (µs) | Wall-clock per launch (µs) | 1-launch command buffer, wall-clock (µs) |
|---|---|---|---|
| concurrent encoder, a barrier before every launch | 0.89 | 1.19 | 191 |
| concurrent encoder, no barriers | 0.07 | 0.18 | 184 |
| serial encoder (Metal orders every launch) | 0.73 | 0.88 | 179 |

| Encoding | 1 launches: GPU / wall (µs) | 16 launches: GPU / wall (µs) | 256 launches: GPU / wall (µs) | 1024 launches: GPU / wall (µs) |
|---|---|---|---|---|
| dependent | 6 / 191 | 58 / 283 | 231 / 506 | 930 / 1447 |
| independent | 2 / 184 | 3 / 185 | 20 / 250 | 78 / 369 |
| serial | 2 / 179 | 13 / 210 | 174 / 391 | 751 / 1083 |
