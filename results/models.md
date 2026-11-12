# Models

**Provisional:** measured while the machine was busy (load average 7.64 at the start). Re-run on a quiet machine for final numbers.

Measured 2026-11-12T06:36:08Z on Apple M4 (4 performance cores, 10 cores, 16 GB), Darwin 27.0.0, commit `b61e64e`. Reproduce with `tessera bench models`.

### resnet50

| system | ms, median (p10 to p90) | vs ONNX Runtime | max abs err | max rel err | top-1 | notes |
|---|---|---|---|---|---|---|
| onnxruntime-cpu | 23.03 (22.15 to 23.70) | 1.00x | 0.00e+00 | 0.00e+00 | 100% | default threads |
| pytorch-cpu | 24.07 (21.32 to 26.28) | 0.96x | 4.77e-06 | 1.03e-06 | 100% | 4 threads |
| pytorch-mps | 23.21 (20.95 to 24.71) | 0.99x | 5.25e-06 | 1.02e-06 | 100% |  |
| tessera-c:sme-untuned | 76.18 (74.18 to 77.39) | 0.30x | 3.28e-06 | 6.53e-07 | 100% | 4 threads, 56 kernels |
| tessera-metal-untuned | 64.52 (62.36 to 65.84) | 0.36x | 3.07e-06 | 6.06e-07 | 100% | 56 kernels |

Ablations of resnet50 (untuned):

| target | pipeline | kernels | bytes moved (MiB) | arena (MiB) | intermediates (MiB) | ms (median) | runs |
|---|---|---|---|---|---|---|---|
| c:sme | full | 56 | 205.7 | 10.0 | 43.2 | 57.75 | 50 |
| c:sme | fusion off | 123 | 322.3 | 11.5 | 101.5 | 64.29 | 50 |
| c:sme | graph rewrites off | 56 | 205.7 | 10.0 | 43.2 | 57.76 | 50 |
| c:sme | no schedule | 56 | 205.7 | 10.0 | 43.2 | 4719.28 | 12 |
| metal | full | 56 | 205.7 | 10.0 | 43.2 | 64.51 | 50 |
| metal | fusion off | 123 | 322.3 | 11.5 | 101.5 | 65.25 | 50 |
| metal | graph rewrites off | 56 | 205.7 | 10.0 | 43.2 | 63.78 | 50 |

Measured 2026-11-12T06:06:02Z at commit `bad21ec`, load average 5.44; 2026-11-12T06:33:13Z at commit `6e6e424`, load average 5.77 (provisional).

### mobilenet_v2

| system | ms, median (p10 to p90) | vs ONNX Runtime | max abs err | max rel err | top-1 | notes |
|---|---|---|---|---|---|---|
| onnxruntime-cpu | 4.06 (3.16 to 4.37) | 1.00x | 0.00e+00 | 0.00e+00 | 100% | default threads |
| pytorch-cpu | 38.13 (33.44 to 43.51) | 0.11x | 1.38e-05 | 3.99e-06 | 100% | 4 threads |
| pytorch-mps | 13.49 (13.12 to 13.88) | 0.30x | 1.10e-05 | 2.83e-06 | 100% |  |
| tessera-c:sme-untuned | 5.92 (5.71 to 6.39) | 0.69x | 9.06e-06 | 2.51e-06 | 100% | 4 threads, 54 kernels |
| tessera-metal-untuned | 6.36 (6.20 to 6.73) | 0.64x | 8.11e-06 | 2.24e-06 | 100% | 54 kernels |

Ablations of mobilenet_v2 (untuned):

| target | pipeline | kernels | bytes moved (MiB) | arena (MiB) | intermediates (MiB) | ms (median) | runs |
|---|---|---|---|---|---|---|---|
| c:sme | full | 54 | 65.7 | 6.5 | 25.5 | 7.65 | 50 |
| c:sme | fusion off | 101 | 115.1 | 10.0 | 50.2 | 7.40 | 50 |
| c:sme | graph rewrites off | 54 | 65.7 | 6.5 | 25.5 | 5.35 | 50 |
| c:sme | no schedule | 54 | 65.7 | 6.5 | 25.5 | 183.10 | 50 |
| metal | full | 54 | 65.7 | 6.5 | 25.5 | 5.82 | 50 |
| metal | fusion off | 101 | 115.1 | 10.0 | 50.2 | 6.13 | 50 |
| metal | graph rewrites off | 54 | 65.7 | 6.5 | 25.5 | 5.77 | 50 |

Measured 2026-11-12T06:05:35Z at commit `bad21ec`, load average 4.18; 2026-11-12T06:32:40Z at commit `6e6e424`, load average 5.46 (provisional).

### bert

| system | ms, median (p10 to p90) | vs ONNX Runtime | max abs err | max rel err | top-1 | notes |
|---|---|---|---|---|---|---|
| onnxruntime-cpu | 36.87 (33.86 to 39.30) | 1.00x | 0.00e+00 | 0.00e+00 |  | default threads |
| pytorch-cpu | 54.85 (49.42 to 56.34) | 0.67x | 2.12e-05 | 2.44e-06 |  | 4 threads |
| pytorch-mps | 22.44 (21.00 to 24.00) | 1.64x | 8.64e-06 | 9.95e-07 |  |  |
| tessera-c:sme-untuned | 144.25 (140.33 to 154.42) | 0.26x | 1.54e-05 | 1.77e-06 |  | 4 threads, 134 kernels |
| tessera-metal-untuned | 48.77 (46.79 to 50.19) | 0.76x | 1.65e-05 | 1.89e-06 |  | 134 kernels |

Ablations of bert (untuned):

| target | pipeline | kernels | bytes moved (MiB) | arena (MiB) | intermediates (MiB) | ms (median) | runs |
|---|---|---|---|---|---|---|---|
| c:sme | full | 134 | 577.8 | 2.3 | 72.1 | 159.10 | 50 |
| c:sme | fusion off | 320 | 787.1 | 3.8 | 176.7 | 150.43 | 50 |
| c:sme | graph rewrites off | 146 | 631.8 | 3.8 | 90.1 | 165.77 | 50 |
| c:sme | no schedule | 134 | 577.8 | 2.3 | 72.1 | 17130.38 | 3 |
| metal | full | 134 | 577.8 | 2.3 | 72.1 | 48.93 | 50 |
| metal | fusion off | 320 | 787.1 | 3.8 | 176.7 | 52.26 | 50 |
| metal | graph rewrites off | 146 | 631.8 | 3.8 | 90.1 | 53.87 | 50 |

Measured 2026-11-12T06:07:37Z at commit `bad21ec`, load average 6.43; 2026-11-12T06:34:10Z at commit `b61e64e`, load average 6.57 (provisional).

### gpt2

| system | ms, median (p10 to p90) | vs ONNX Runtime | max abs err | max rel err | top-1 | notes |
|---|---|---|---|---|---|---|
| onnxruntime-cpu | 50.18 (47.62 to 59.81) | 1.00x | 0.00e+00 | 0.00e+00 |  | default threads |
| pytorch-cpu | 70.97 (61.95 to 74.27) | 0.71x | 2.59e-04 | 9.64e-07 |  | 4 threads |
| pytorch-mps | 39.10 (30.62 to 46.02) | 1.28x | 2.44e-04 | 9.08e-07 |  |  |
| tessera-c:sme-untuned | 324.91 (310.40 to 452.98) | 0.15x | 3.20e-04 | 1.19e-06 |  | 4 threads, 111 kernels |
| tessera-metal-untuned | 121.13 (99.55 to 140.99) | 0.41x | 3.05e-04 | 1.13e-06 |  | 111 kernels |

Ablations of gpt2 (untuned):

| target | pipeline | kernels | bytes moved (MiB) | arena (MiB) | intermediates (MiB) | ms (median) | runs |
|---|---|---|---|---|---|---|---|
| c:sme | no schedule | 111 | 812.6 | 3.4 | 72.8 | 23166.59 | 3 |
| c:sme | full | 111 | 812.6 | 3.4 | 72.8 | 407.84 | 50 |
| c:sme | fusion off | 304 | 1060.9 | 3.8 | 190.1 | 355.44 | 50 |
| c:sme | graph rewrites off | 134 | 879.4 | 5.6 | 94.9 | 385.54 | 50 |
| metal | full | 111 | 812.6 | 3.4 | 72.8 | 100.85 | 50 |
| metal | fusion off | 304 | 1060.9 | 3.8 | 190.1 | 99.57 | 50 |
| metal | graph rewrites off | 134 | 879.4 | 5.6 | 94.9 | 87.06 | 50 |

Measured 2026-11-12T06:09:19Z at commit `bad21ec`, load average 5.99; 2026-11-12T06:36:08Z at commit `b61e64e`, load average 7.64 (provisional).
