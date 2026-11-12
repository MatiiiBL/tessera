# ResNet-50 convolutions

The distinct convolutions of ResNet-50 at 224 x 224, batch 1, NHWC input and HWIO weights, each with its bias and ReLU, as the conv2d template schedules them for each target.

Measured 2026-11-12T07:22:56+00:00 on Apple M4 (4 performance cores), Darwin 27.0.0, Apple clang version 21.0.0 (clang-2100.0.123.102), NumPy 2.5.4; load average 2.44 at the start. Throughput in GFLOPS (higher is better).

| case | generic t1 | generic t4 | neon t1 | neon t4 | sme t1 | sme t4 |
|---|---:|---:|---:|---:|---:|---:|
| conv1 | 110 | 352 | 79 | 353 | 211 | 205 |
| layer1 1x1 reduce | 105 | 312 | 93 | 313 | 329 | 221 |
| layer1 3x3 | 102 | 334 | 101 | 336 | 649 | 495 |
| layer1 1x1 expand | 101 | 343 | 104 | 293 | 484 | 353 |
| layer1 1x1 reduce (256) | 106 | 321 | 97 | 321 | 517 | 316 |
| layer2 1x1 reduce | 106 | 333 | 109 | 292 | 731 | 589 |
| layer2 3x3 stride 2 | 97 | 305 | 112 | 300 | 776 | 708 |
| layer2 1x1 expand | 101 | 321 | 110 | 347 | 601 | 490 |
| layer2 downsample | 109 | 336 | 113 | 365 | 825 | 558 |
| layer2 1x1 reduce (512) | 91 | 337 | 110 | 297 | 865 | 540 |
| layer2 3x3 | 99 | 311 | 113 | 304 | 884 | 768 |
| layer3 1x1 reduce | 104 | 333 | 111 | 324 | 1106 | 648 |
| layer3 3x3 stride 2 | 61 | 200 | 112 | 323 | 1020 | 656 |
| layer3 1x1 expand | 64 | 213 | 110 | 364 | 717 | 515 |
| layer3 downsample | 65 | 212 | 112 | 375 | 919 | 554 |
| layer3 1x1 reduce (1024) | 64 | 212 | 114 | 322 | 962 | 627 |
| layer3 3x3 | 62 | 200 | 115 | 322 | 1090 | 670 |
| layer4 1x1 reduce | 63 | 208 | 111 | 363 | 926 | 526 |
| layer4 3x3 stride 2 | 72 | 237 | 101 | 334 | 810 | 457 |
| layer4 1x1 expand | 92 | 271 | 101 | 336 | 751 | 491 |
| layer4 downsample | 67 | 235 | 100 | 339 | 797 | 501 |
| layer4 1x1 reduce (2048) | 69 | 241 | 103 | 336 | 853 | 453 |
| layer4 3x3 | 84 | 268 | 101 | 336 | 841 | 466 |

Reproduce: `uv run python -m tessera.backend.c.bench`.
