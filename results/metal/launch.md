# Metal: the cost of a kernel launch

**Provisional:** measured while the machine was busy (load average 6.28 at the start). Re-run on a quiet machine for final numbers.

Measured 2026-11-12T05:54:24Z on Apple M4 (4 performance cores, 10 cores, 16 GB), Darwin 27.0.0, commit `e854764`. Reproduce with `uv run python -m tessera.backend.metal.bench launch`.

A one-threadgroup kernel that updates one float, launched 1 to 1024 times in one
command buffer. Per-launch costs are the slope of the median time (over 30 runs) against the
number of launches; the last column is the round trip of a command buffer with one
launch, from encoding to the CPU seeing it complete.

| Encoding | GPU per launch (µs) | Wall-clock per launch (µs) | 1-launch command buffer, wall-clock (µs) |
|---|---|---|---|
| concurrent encoder, a barrier before every launch | 3.19 | 3.65 | 181 |
| concurrent encoder, no barriers | 0.18 | 0.36 | 181 |
| serial encoder (Metal orders every launch) | 2.82 | 3.05 | 176 |

| Encoding | 1 launches: GPU / wall (µs) | 16 launches: GPU / wall (µs) | 256 launches: GPU / wall (µs) | 1024 launches: GPU / wall (µs) |
|---|---|---|---|---|
| dependent | 6 / 181 | 53 / 291 | 818 / 1142 | 3267 / 3940 |
| independent | 6 / 181 | 8 / 187 | 53 / 310 | 194 / 549 |
| serial | 6 / 176 | 48 / 259 | 717 / 1015 | 2888 / 3320 |
