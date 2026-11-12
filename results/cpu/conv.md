# ResNet-50 convolutions

The distinct convolutions of ResNet-50 at 224 x 224, batch 1, NHWC input and HWIO weights, each with its bias and ReLU, as the conv2d template schedules them for each target.

Measured 2026-11-12T12:04:37+00:00 on Apple M4 (4 performance cores), Darwin 27.0.0, Apple clang version 21.0.0 (clang-2100.0.123.102), NumPy 2.5.4; load average 2.83 at the start. Throughput in GFLOPS (higher is better).

| case | generic t1 | generic t4 | neon t1 | neon t4 | sme t1 | sme t4 |
|---|---:|---:|---:|---:|---:|---:|
| conv1 | 116 | 377 | 79 | 377 | 225 | 223 |
| layer1 1x1 reduce | 113 | 352 | 93 | 354 | 330 | 252 |
| layer1 3x3 | 116 | 351 | 102 | 354 | 651 | 565 |
| layer1 1x1 expand | 113 | 364 | 107 | 313 | 483 | 684 |
| layer1 1x1 reduce (256) | 106 | 352 | 100 | 350 | 517 | 477 |
| layer2 1x1 reduce | 115 | 358 | 113 | 308 | 727 | 931 |
| layer2 3x3 stride 2 | 101 | 331 | 115 | 314 | 790 | 759 |
| layer2 1x1 expand | 111 | 351 | 111 | 363 | 602 | 1051 |
| layer2 downsample | 113 | 359 | 115 | 378 | 820 | 1222 |
| layer2 1x1 reduce (512) | 116 | 368 | 114 | 315 | 866 | 853 |
| layer2 3x3 | 106 | 340 | 115 | 328 | 877 | 856 |
| layer3 1x1 reduce | 113 | 362 | 117 | 339 | 1104 | 1164 |
| layer3 3x3 stride 2 | 64 | 214 | 116 | 338 | 1021 | 1003 |
| layer3 1x1 expand | 66 | 226 | 111 | 379 | 717 | 965 |
| layer3 downsample | 65 | 229 | 114 | 392 | 919 | 1294 |
| layer3 1x1 reduce (1024) | 64 | 226 | 115 | 335 | 963 | 925 |
| layer3 3x3 | 65 | 213 | 116 | 342 | 1083 | 1072 |
| layer4 1x1 reduce | 64 | 225 | 115 | 383 | 928 | 1109 |
| layer4 3x3 stride 2 | 72 | 254 | 104 | 351 | 808 | 910 |
| layer4 1x1 expand | 92 | 317 | 102 | 346 | 746 | 1132 |
| layer4 downsample | 67 | 254 | 102 | 354 | 808 | 1177 |
| layer4 1x1 reduce (2048) | 69 | 258 | 104 | 352 | 847 | 1078 |
| layer4 3x3 | 81 | 286 | 104 | 351 | 838 | 843 |

Reproduce: `uv run python -m tessera.backend.c.bench`.
