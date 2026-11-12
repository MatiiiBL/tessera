# Models

**Provisional:** measured while the machine was busy (load average 5.99 at the start). Re-run on a quiet machine for final numbers.

Measured 2026-11-12T06:09:19Z on Apple M4 (4 performance cores, 10 cores, 16 GB), Darwin 27.0.0, commit `bad21ec`. Reproduce with `tessera bench models`.

### resnet50

| system | ms, median (p10 to p90) | vs ONNX Runtime | max abs err | max rel err | top-1 | notes |
|---|---|---|---|---|---|---|
| onnxruntime-cpu | 31.90 (28.26 to 37.63) | 1.00x | 0.00e+00 | 0.00e+00 | 100% | default threads |
| pytorch-cpu | 38.17 (34.61 to 51.59) | 0.84x | 4.77e-06 | 1.03e-06 | 100% | 4 threads |
| pytorch-mps | 35.80 (29.46 to 50.75) | 0.89x | 5.25e-06 | 1.02e-06 | 100% |  |

Ablations of resnet50 (untuned):

| target | pipeline | kernels | bytes moved (MiB) | arena (MiB) | intermediates (MiB) | ms (median) | runs |
|---|---|---|---|---|---|---|---|
| c:sme | no schedule | 56 | 205.7 | 10.0 | 43.2 | 4719.28 | 12 |

Measured 2026-11-12T06:06:02Z at commit `bad21ec`, load average 5.44 (provisional).

### mobilenet_v2

| system | ms, median (p10 to p90) | vs ONNX Runtime | max abs err | max rel err | top-1 | notes |
|---|---|---|---|---|---|---|
| onnxruntime-cpu | 4.84 (4.19 to 5.63) | 1.00x | 0.00e+00 | 0.00e+00 | 100% | default threads |
| pytorch-cpu | 36.91 (34.73 to 39.05) | 0.13x | 1.38e-05 | 3.99e-06 | 100% | 4 threads |
| pytorch-mps | 23.53 (16.39 to 28.79) | 0.21x | 1.10e-05 | 2.83e-06 | 100% |  |

Ablations of mobilenet_v2 (untuned):

| target | pipeline | kernels | bytes moved (MiB) | arena (MiB) | intermediates (MiB) | ms (median) | runs |
|---|---|---|---|---|---|---|---|
| c:sme | no schedule | 54 | 65.7 | 6.5 | 25.5 | 183.10 | 50 |

Measured 2026-11-12T06:05:35Z at commit `bad21ec`, load average 4.18 (provisional).

### bert

| system | ms, median (p10 to p90) | vs ONNX Runtime | max abs err | max rel err | top-1 | notes |
|---|---|---|---|---|---|---|
| onnxruntime-cpu | 47.06 (43.47 to 59.63) | 1.00x | 0.00e+00 | 0.00e+00 |  | default threads |
| pytorch-cpu | 63.42 (51.41 to 69.08) | 0.74x | 2.12e-05 | 2.44e-06 |  | 4 threads |
| pytorch-mps | 21.06 (20.42 to 21.97) | 2.23x | 8.64e-06 | 9.95e-07 |  |  |

Ablations of bert (untuned):

| target | pipeline | kernels | bytes moved (MiB) | arena (MiB) | intermediates (MiB) | ms (median) | runs |
|---|---|---|---|---|---|---|---|
| c:sme | no schedule | 134 | 577.8 | 2.3 | 72.1 | 17130.38 | 3 |

Measured 2026-11-12T06:07:37Z at commit `bad21ec`, load average 6.43 (provisional).

### gpt2

| system | ms, median (p10 to p90) | vs ONNX Runtime | max abs err | max rel err | top-1 | notes |
|---|---|---|---|---|---|---|
| onnxruntime-cpu | 65.75 (61.29 to 68.55) | 1.00x | 0.00e+00 | 0.00e+00 |  | default threads |
| pytorch-cpu | 85.48 (82.27 to 94.60) | 0.77x | 2.59e-04 | 9.64e-07 |  | 4 threads |
| pytorch-mps | 40.42 (34.71 to 47.42) | 1.63x | 2.44e-04 | 9.08e-07 |  |  |

Ablations of gpt2 (untuned):

| target | pipeline | kernels | bytes moved (MiB) | arena (MiB) | intermediates (MiB) | ms (median) | runs |
|---|---|---|---|---|---|---|---|
| c:sme | no schedule | 111 | 812.6 | 3.4 | 72.8 | 23166.59 | 3 |

Measured 2026-11-12T06:09:19Z at commit `bad21ec`, load average 5.99 (provisional).
