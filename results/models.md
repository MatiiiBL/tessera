# Models

Measured 2026-11-12T11:00:56Z on Apple M4 (4 performance cores, 10 cores, 16 GB), Darwin 27.0.0, commit `beb8516`. Reproduce with `tessera bench models`.

### resnet50

| system | ms, median (p10 to p90) | vs ONNX Runtime | max abs err | max rel err | top-1 | notes |
|---|---|---|---|---|---|---|
| onnxruntime-cpu | 11.27 (11.19 to 11.35) | 1.00x | 0.00e+00 | 0.00e+00 | 100% | default threads |
| pytorch-cpu | 13.12 (13.01 to 13.35) | 0.86x | 4.77e-06 | 1.03e-06 | 100% | 4 threads |
| pytorch-mps | 12.29 (12.18 to 12.86) | 0.92x | 5.25e-06 | 1.02e-06 | 100% |  |
| tessera-c:sme | 12.52 (12.34 to 12.60) | 0.90x | 3.10e-06 | 6.06e-07 | 100% | 4 threads, 56 kernels, 56 tuned |
| tessera-c:sme-untuned | 20.23 (17.96 to 24.71) | 0.56x | 3.28e-06 | 6.53e-07 | 100% | 4 threads, 56 kernels |
| tessera-c:neon-untuned | 26.11 (25.76 to 26.94) | 0.43x | 3.28e-06 | 6.53e-07 | 100% | 4 threads, 56 kernels |
| tessera-metal | 10.74 (10.65 to 10.88) | 1.05x | 3.07e-06 | 6.06e-07 | 100% | 56 kernels, 56 tuned |
| tessera-metal-untuned | 16.94 (16.83 to 17.11) | 0.67x | 3.07e-06 | 6.06e-07 | 100% | 56 kernels |

Ablations of resnet50 (untuned):

| target | pipeline | kernels | bytes moved (MiB) | arena (MiB) | intermediates (MiB) | ms (median) | runs |
|---|---|---|---|---|---|---|---|
| c:sme | full | 56 | 205.7 | 10.0 | 43.2 | 24.78 | 50 |
| c:sme | fusion off | 123 | 322.3 | 11.5 | 101.5 | 19.96 | 50 |
| c:sme | graph rewrites off | 56 | 205.7 | 10.0 | 43.2 | 24.33 | 50 |
| c:sme | no schedule | 56 | 205.7 | 10.0 | 43.2 | 2946.94 | 20 |
| metal | full | 56 | 205.7 | 10.0 | 43.2 | 16.95 | 50 |
| metal | fusion off | 123 | 322.3 | 11.5 | 101.5 | 17.46 | 50 |
| metal | graph rewrites off | 56 | 205.7 | 10.0 | 43.2 | 16.90 | 50 |
| metal | no schedule |  |  |  |  |  | not measured: unscheduled Metal kernels run on one GPU thread |

Measured 2026-11-12T10:56:21Z at commit `beb8516`, load average 2.28.

### mobilenet_v2

| system | ms, median (p10 to p90) | vs ONNX Runtime | max abs err | max rel err | top-1 | notes |
|---|---|---|---|---|---|---|
| onnxruntime-cpu | 2.38 (2.33 to 2.52) | 1.00x | 0.00e+00 | 0.00e+00 | 100% | default threads |
| pytorch-cpu | 15.15 (14.80 to 15.53) | 0.16x | 1.38e-05 | 3.99e-06 | 100% | 4 threads |
| pytorch-mps | 6.66 (6.42 to 6.87) | 0.36x | 1.10e-05 | 2.83e-06 | 100% |  |
| tessera-c:sme | 1.90 (1.89 to 2.03) | 1.25x | 7.39e-06 | 1.93e-06 | 100% | 4 threads, 54 kernels, 54 tuned |
| tessera-c:sme-untuned | 3.35 (3.25 to 3.44) | 0.71x | 9.06e-06 | 2.51e-06 | 100% | 4 threads, 54 kernels |
| tessera-c:neon-untuned | 2.89 (2.84 to 2.97) | 0.82x | 9.06e-06 | 2.51e-06 | 100% | 4 threads, 54 kernels |
| tessera-metal | 1.76 (1.73 to 1.82) | 1.36x | 8.11e-06 | 2.24e-06 | 100% | 54 kernels, 54 tuned |
| tessera-metal-untuned | 1.96 (1.92 to 2.02) | 1.22x | 8.11e-06 | 2.24e-06 | 100% | 54 kernels |

Ablations of mobilenet_v2 (untuned):

| target | pipeline | kernels | bytes moved (MiB) | arena (MiB) | intermediates (MiB) | ms (median) | runs |
|---|---|---|---|---|---|---|---|
| c:sme | full | 54 | 65.7 | 6.5 | 25.5 | 3.34 | 50 |
| c:sme | fusion off | 101 | 115.1 | 10.0 | 50.2 | 3.17 | 50 |
| c:sme | graph rewrites off | 54 | 65.7 | 6.5 | 25.5 | 3.33 | 50 |
| c:sme | no schedule | 54 | 65.7 | 6.5 | 25.5 | 88.48 | 50 |
| metal | full | 54 | 65.7 | 6.5 | 25.5 | 1.95 | 50 |
| metal | fusion off | 101 | 115.1 | 10.0 | 50.2 | 2.31 | 50 |
| metal | graph rewrites off | 54 | 65.7 | 6.5 | 25.5 | 1.94 | 50 |
| metal | no schedule |  |  |  |  |  | not measured: unscheduled Metal kernels run on one GPU thread |

Measured 2026-11-12T10:58:17Z at commit `beb8516`, load average 1.91.

### bert

| system | ms, median (p10 to p90) | vs ONNX Runtime | max abs err | max rel err | top-1 | notes |
|---|---|---|---|---|---|---|
| onnxruntime-cpu | 20.68 (20.32 to 20.97) | 1.00x | 0.00e+00 | 0.00e+00 |  | default threads |
| pytorch-cpu | 26.53 (26.36 to 27.01) | 0.78x | 2.12e-05 | 2.44e-06 |  | 4 threads |
| pytorch-mps | 14.85 (14.61 to 15.02) | 1.39x | 8.64e-06 | 9.95e-07 |  |  |
| tessera-c:sme | 27.85 (27.73 to 28.44) | 0.74x | 1.54e-05 | 1.77e-06 |  | 4 threads, 134 kernels, 134 tuned |
| tessera-c:sme-untuned | 27.57 (27.16 to 27.83) | 0.75x | 1.54e-05 | 1.77e-06 |  | 4 threads, 134 kernels |
| tessera-c:neon-untuned | 64.22 (63.05 to 64.66) | 0.32x | 1.54e-05 | 1.77e-06 |  | 4 threads, 134 kernels |
| tessera-metal | 11.78 (11.64 to 11.88) | 1.76x | 1.41e-05 | 1.62e-06 |  | 134 kernels, 134 tuned |
| tessera-metal-untuned | 12.50 (12.35 to 12.60) | 1.65x | 1.65e-05 | 1.89e-06 |  | 134 kernels |

Ablations of bert (untuned):

| target | pipeline | kernels | bytes moved (MiB) | arena (MiB) | intermediates (MiB) | ms (median) | runs |
|---|---|---|---|---|---|---|---|
| c:sme | full | 134 | 577.8 | 2.3 | 72.1 | 27.96 | 50 |
| c:sme | fusion off | 320 | 787.1 | 3.8 | 176.7 | 23.21 | 50 |
| c:sme | graph rewrites off | 146 | 631.8 | 3.8 | 90.1 | 60.41 | 50 |
| c:sme | no schedule | 134 | 577.8 | 2.3 | 72.1 | 8927.49 | 6 |
| metal | full | 134 | 577.8 | 2.3 | 72.1 | 12.29 | 50 |
| metal | fusion off | 320 | 787.1 | 3.8 | 176.7 | 11.09 | 50 |
| metal | graph rewrites off | 146 | 631.8 | 3.8 | 90.1 | 21.62 | 50 |
| metal | no schedule |  |  |  |  |  | not measured: unscheduled Metal kernels run on one GPU thread |

Measured 2026-11-12T10:58:45Z at commit `beb8516`, load average 1.94.

### gpt2

| system | ms, median (p10 to p90) | vs ONNX Runtime | max abs err | max rel err | top-1 | notes |
|---|---|---|---|---|---|---|
| onnxruntime-cpu | 29.68 (28.65 to 30.12) | 1.00x | 0.00e+00 | 0.00e+00 |  | default threads |
| pytorch-cpu | 39.11 (38.88 to 40.96) | 0.76x | 2.59e-04 | 9.64e-07 |  | 4 threads |
| pytorch-mps | 22.27 (21.96 to 23.07) | 1.33x | 2.44e-04 | 9.08e-07 |  |  |
| tessera-c:sme | 56.64 (53.32 to 59.05) | 0.52x | 2.75e-04 | 1.02e-06 |  | 4 threads, 111 kernels, 111 tuned |
| tessera-c:sme-untuned | 47.36 (47.13 to 47.82) | 0.63x | 2.75e-04 | 1.02e-06 |  | 4 threads, 111 kernels |
| tessera-c:neon-untuned | 95.38 (93.74 to 103.26) | 0.31x | 2.75e-04 | 1.02e-06 |  | 4 threads, 111 kernels |
| tessera-metal | 15.78 (15.44 to 16.27) | 1.88x | 2.52e-04 | 9.36e-07 |  | 111 kernels, 111 tuned |
| tessera-metal-untuned | 16.81 (16.64 to 16.94) | 1.77x | 3.05e-04 | 1.13e-06 |  | 111 kernels |

Ablations of gpt2 (untuned):

| target | pipeline | kernels | bytes moved (MiB) | arena (MiB) | intermediates (MiB) | ms (median) | runs |
|---|---|---|---|---|---|---|---|
| c:sme | full | 111 | 812.6 | 3.4 | 72.8 | 47.28 | 50 |
| c:sme | fusion off | 304 | 1060.9 | 3.8 | 190.1 | 34.15 | 50 |
| c:sme | graph rewrites off | 134 | 879.4 | 5.6 | 94.9 | 78.40 | 50 |
| c:sme | no schedule | 111 | 812.6 | 3.4 | 72.8 | 11058.09 | 5 |
| metal | full | 111 | 812.6 | 3.4 | 72.8 | 16.62 | 50 |
| metal | fusion off | 304 | 1060.9 | 3.8 | 190.1 | 15.37 | 50 |
| metal | graph rewrites off | 134 | 879.4 | 5.6 | 94.9 | 21.64 | 50 |
| metal | no schedule |  |  |  |  |  | not measured: unscheduled Metal kernels run on one GPU thread |

Measured 2026-11-12T11:00:56Z at commit `beb8516`, load average 2.33.
