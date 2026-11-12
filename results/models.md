# Models

Measured 2026-11-12T13:49:45Z on Apple M4 (4 performance cores, 10 cores, 16 GB), Darwin 27.0.0, commit `e047e7f`. Reproduce with `tessera bench models`.

### resnet50

| system | ms, median (p10 to p90) | vs ONNX Runtime | max abs err | max rel err | top-1 | notes |
|---|---|---|---|---|---|---|
| onnxruntime-cpu | 12.14 (12.08 to 12.24) | 1.00x | 0.00e+00 | 0.00e+00 | 100% | default threads |
| pytorch-cpu | 13.18 (13.08 to 13.53) | 0.92x | 4.77e-06 | 1.03e-06 | 100% | 4 threads |
| pytorch-mps | 12.28 (12.19 to 12.38) | 0.99x | 5.25e-06 | 1.02e-06 | 100% |  |
| tessera-c:sme | 11.52 (11.17 to 15.05) | 1.05x | 3.10e-06 | 6.06e-07 | 100% | 4 threads, 56 kernels, 56 tuned |
| tessera-c:sme-untuned | 12.02 (11.80 to 16.34) | 1.01x | 3.28e-06 | 6.53e-07 | 100% | 4 threads, 56 kernels |
| tessera-c:neon-untuned | 26.03 (25.61 to 26.46) | 0.47x | 3.28e-06 | 6.53e-07 | 100% | 4 threads, 56 kernels |
| tessera-metal | 10.71 (10.66 to 10.79) | 1.13x | 3.07e-06 | 6.06e-07 | 100% | 56 kernels, 56 tuned |
| tessera-metal-untuned | 16.92 (16.85 to 17.04) | 0.72x | 3.07e-06 | 6.06e-07 | 100% | 56 kernels |

Ablations of resnet50 (untuned):

| target | pipeline | kernels | bytes moved (MiB) | arena (MiB) | intermediates (MiB) | ms (median) | runs |
|---|---|---|---|---|---|---|---|
| c:sme | full | 56 | 205.7 | 10.0 | 43.2 | 15.49 | 50 |
| c:sme | fusion off | 123 | 322.3 | 11.5 | 101.5 | 12.07 | 50 |
| c:sme | graph rewrites off | 56 | 205.7 | 10.0 | 43.2 | 11.99 | 50 |
| c:sme | no schedule | 56 | 205.7 | 10.0 | 43.2 | 2936.76 | 20 |
| metal | full | 56 | 205.7 | 10.0 | 43.2 | 16.97 | 50 |
| metal | fusion off | 123 | 322.3 | 11.5 | 101.5 | 17.37 | 50 |
| metal | graph rewrites off | 56 | 205.7 | 10.0 | 43.2 | 16.91 | 50 |
| metal | no schedule |  |  |  |  |  | not measured: unscheduled Metal kernels run on one GPU thread |

Measured 2026-11-12T13:43:48Z at commit `e047e7f`, load average 1.42.

### mobilenet_v2

| system | ms, median (p10 to p90) | vs ONNX Runtime | max abs err | max rel err | top-1 | notes |
|---|---|---|---|---|---|---|
| onnxruntime-cpu | 2.43 (2.35 to 2.56) | 1.00x | 0.00e+00 | 0.00e+00 | 100% | default threads |
| pytorch-cpu | 15.25 (14.82 to 15.37) | 0.16x | 1.38e-05 | 3.99e-06 | 100% | 4 threads |
| pytorch-mps | 6.62 (6.49 to 6.73) | 0.37x | 1.10e-05 | 2.83e-06 | 100% |  |
| tessera-c:sme | 2.15 (2.12 to 2.17) | 1.13x | 8.46e-06 | 2.51e-06 | 100% | 4 threads, 54 kernels, 54 tuned |
| tessera-c:sme-untuned | 3.15 (3.04 to 3.29) | 0.77x | 9.06e-06 | 2.51e-06 | 100% | 4 threads, 54 kernels |
| tessera-c:neon-untuned | 3.01 (2.90 to 3.17) | 0.81x | 9.06e-06 | 2.51e-06 | 100% | 4 threads, 54 kernels |
| tessera-metal | 1.74 (1.70 to 1.79) | 1.40x | 8.11e-06 | 2.24e-06 | 100% | 54 kernels, 54 tuned |
| tessera-metal-untuned | 1.95 (1.85 to 2.02) | 1.25x | 8.11e-06 | 2.24e-06 | 100% | 54 kernels |

Ablations of mobilenet_v2 (untuned):

| target | pipeline | kernels | bytes moved (MiB) | arena (MiB) | intermediates (MiB) | ms (median) | runs |
|---|---|---|---|---|---|---|---|
| c:sme | full | 54 | 65.7 | 6.5 | 25.5 | 3.18 | 50 |
| c:sme | fusion off | 101 | 115.1 | 10.0 | 50.2 | 3.29 | 50 |
| c:sme | graph rewrites off | 54 | 65.7 | 6.5 | 25.5 | 3.18 | 50 |
| c:sme | no schedule | 54 | 65.7 | 6.5 | 25.5 | 89.24 | 50 |
| metal | full | 54 | 65.7 | 6.5 | 25.5 | 1.94 | 50 |
| metal | fusion off | 101 | 115.1 | 10.0 | 50.2 | 2.30 | 50 |
| metal | graph rewrites off | 54 | 65.7 | 6.5 | 25.5 | 1.88 | 50 |
| metal | no schedule |  |  |  |  |  | not measured: unscheduled Metal kernels run on one GPU thread |

Measured 2026-11-12T13:46:01Z at commit `e047e7f`, load average 2.12.

### bert

| system | ms, median (p10 to p90) | vs ONNX Runtime | max abs err | max rel err | top-1 | notes |
|---|---|---|---|---|---|---|
| onnxruntime-cpu | 21.97 (21.87 to 22.23) | 1.00x | 0.00e+00 | 0.00e+00 |  | default threads |
| pytorch-cpu | 26.46 (26.28 to 26.82) | 0.83x | 2.12e-05 | 2.44e-06 |  | 4 threads |
| pytorch-mps | 14.85 (14.61 to 15.29) | 1.48x | 8.64e-06 | 9.95e-07 |  |  |
| tessera-c:sme | 18.06 (17.66 to 20.31) | 1.22x | 1.54e-05 | 1.77e-06 |  | 4 threads, 134 kernels, 134 tuned |
| tessera-c:sme-untuned | 21.98 (21.88 to 22.07) | 1.00x | 1.54e-05 | 1.77e-06 |  | 4 threads, 134 kernels |
| tessera-c:neon-untuned | 60.79 (59.53 to 61.84) | 0.36x | 1.54e-05 | 1.77e-06 |  | 4 threads, 134 kernels |
| tessera-metal | 10.31 (10.17 to 10.42) | 2.13x | 1.31e-05 | 1.51e-06 |  | 134 kernels, 134 tuned |
| tessera-metal-untuned | 10.54 (10.42 to 10.68) | 2.08x | 1.65e-05 | 1.89e-06 |  | 134 kernels |

Ablations of bert (untuned):

| target | pipeline | kernels | bytes moved (MiB) | arena (MiB) | intermediates (MiB) | ms (median) | runs |
|---|---|---|---|---|---|---|---|
| c:sme | full | 134 | 577.8 | 2.3 | 72.1 | 22.06 | 50 |
| c:sme | fusion off | 320 | 787.1 | 3.8 | 176.7 | 22.52 | 50 |
| c:sme | graph rewrites off | 146 | 631.8 | 3.8 | 90.1 | 22.94 | 50 |
| c:sme | no schedule | 134 | 577.8 | 2.3 | 72.1 | 9173.33 | 6 |
| metal | full | 134 | 577.8 | 2.3 | 72.1 | 10.37 | 50 |
| metal | fusion off | 320 | 787.1 | 3.8 | 176.7 | 11.31 | 50 |
| metal | graph rewrites off | 146 | 631.8 | 3.8 | 90.1 | 11.75 | 50 |
| metal | no schedule |  |  |  |  |  | not measured: unscheduled Metal kernels run on one GPU thread |

Measured 2026-11-12T13:46:56Z at commit `e047e7f`, load average 2.17.

### gpt2

| system | ms, median (p10 to p90) | vs ONNX Runtime | max abs err | max rel err | top-1 | notes |
|---|---|---|---|---|---|---|
| onnxruntime-cpu | 30.54 (30.30 to 31.56) | 1.00x | 0.00e+00 | 0.00e+00 |  | default threads |
| pytorch-cpu | 39.10 (38.93 to 39.63) | 0.78x | 2.59e-04 | 9.64e-07 |  | 4 threads |
| pytorch-mps | 22.13 (21.91 to 23.34) | 1.38x | 2.44e-04 | 9.08e-07 |  |  |
| tessera-c:sme | 32.68 (31.90 to 33.44) | 0.93x | 2.75e-04 | 1.02e-06 |  | 4 threads, 111 kernels, 111 tuned |
| tessera-c:sme-untuned | 32.36 (32.17 to 34.55) | 0.94x | 2.75e-04 | 1.02e-06 |  | 4 threads, 111 kernels |
| tessera-c:neon-untuned | 94.43 (89.06 to 133.97) | 0.32x | 2.75e-04 | 1.02e-06 |  | 4 threads, 111 kernels |
| tessera-metal | 14.18 (13.65 to 14.92) | 2.15x | 3.97e-04 | 1.47e-06 |  | 111 kernels, 111 tuned |
| tessera-metal-untuned | 14.65 (14.37 to 15.10) | 2.08x | 3.05e-04 | 1.13e-06 |  | 111 kernels |

Ablations of gpt2 (untuned):

| target | pipeline | kernels | bytes moved (MiB) | arena (MiB) | intermediates (MiB) | ms (median) | runs |
|---|---|---|---|---|---|---|---|
| c:sme | full | 111 | 812.6 | 3.4 | 72.8 | 32.39 | 50 |
| c:sme | fusion off | 304 | 1060.9 | 3.8 | 190.1 | 32.42 | 50 |
| c:sme | graph rewrites off | 134 | 879.4 | 5.6 | 94.9 | 39.42 | 50 |
| c:sme | no schedule | 111 | 812.6 | 3.4 | 72.8 | 11215.88 | 5 |
| metal | full | 111 | 812.6 | 3.4 | 72.8 | 14.53 | 50 |
| metal | fusion off | 304 | 1060.9 | 3.8 | 190.1 | 15.46 | 50 |
| metal | graph rewrites off | 134 | 879.4 | 5.6 | 94.9 | 15.70 | 50 |
| metal | no schedule |  |  |  |  |  | not measured: unscheduled Metal kernels run on one GPU thread |

Measured 2026-11-12T13:49:45Z at commit `e047e7f`, load average 2.2.
