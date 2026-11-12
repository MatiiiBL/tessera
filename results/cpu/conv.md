# ResNet-50 convolutions

The distinct convolutions of ResNet-50 at 224 x 224, batch 1, NHWC input and HWIO weights, each with its bias and ReLU, as the conv2d template schedules them for each target.

Measured 2026-11-12T07:00:01+00:00 on Apple M4 (4 performance cores), Darwin 27.0.0, Apple clang version 21.0.0 (clang-2100.0.123.102), NumPy 2.5.4; load average 5.89 at the start. Throughput in GFLOPS (higher is better).

| case | generic t1 | generic t4 | neon t1 | neon t4 | sme t1 | sme t4 |
|---|---:|---:|---:|---:|---:|---:|
| conv1 | 65 | 134 | 41 | 135 | 114 | 85 |
| layer1 1x1 reduce | 60 | 154 | 54 | 165 | 204 | 124 |
| layer1 3x3 | 68 | 193 | 59 | 168 | 303 | 206 |
| layer1 1x1 expand | 59 | 198 | 55 | 143 | 253 | 196 |
| layer1 1x1 reduce (256) | 60 | 152 | 45 | 147 | 259 | 157 |
| layer2 1x1 reduce | 60 | 149 | 58 | 136 | 390 | 289 |
| layer2 3x3 stride 2 | 54 | 158 | 51 | 106 | 347 | 274 |
| layer2 1x1 expand | 58 | 151 | 55 | 134 | 317 | 288 |
| layer2 downsample | 60 | 148 | 59 | 131 | 381 | 267 |
| layer2 1x1 reduce (512) | 52 | 159 | 51 | 124 | 392 | 268 |
| layer2 3x3 | 55 | 160 | 54 | 167 | 440 | 433 |
| layer3 1x1 reduce | 68 | 200 | 59 | 176 | 557 | 256 |
| layer3 3x3 stride 2 | 38 | 113 | 50 | 123 | 387 | 270 |
| layer3 1x1 expand | 39 | 120 | 61 | 189 | 334 | 277 |
| layer3 downsample | 31 | 115 | 48 | 119 | 341 | 246 |
| layer3 1x1 reduce (1024) | 33 | 114 | 58 | 135 | 389 | 248 |
| layer3 3x3 | 34 | 94 | 50 | 121 | 467 | 274 |
| layer4 1x1 reduce | 36 | 114 | 54 | 130 | 407 | 242 |
| layer4 3x3 stride 2 | 38 | 103 | 54 | 190 | 298 | 246 |
| layer4 1x1 expand | 71 | 174 | 48 | 167 | 268 | 243 |
| layer4 downsample | 37 | 136 | 51 | 176 | 261 | 195 |
| layer4 1x1 reduce (2048) | 40 | 140 | 49 | 156 | 235 | 225 |
| layer4 3x3 | 42 | 146 | 48 | 165 | 273 | 214 |

Reproduce: `uv run python -m tessera.backend.c.bench`.
