# Tuning: random against model-guided search

Measured 2026-11-12T14:11:13Z on Apple M4 (4 performance cores, 10 cores, 16 GB), Darwin 27.0.0, commit `099844c`. Reproduce with `tessera bench tuning --trials 256 --seeds 5 --real-tasks 2 --real-trials 96 --real-seeds 3 --target c:sme`.


Each cell is the best throughput found so far relative to the reference (1.000 is the reference), after that many measurements.

## Synthetic tasks (analytic cost, exact optimum)

**synthetic-matmul:64x64x64** (reference: optimum, 3.0 µs; mean over 5 seeds, worst seed in parentheses)

| trials | random | model |
|---|---|---|
| 8 | 0.293 (0.185) | 0.293 (0.185) |
| 16 | 0.536 (0.366) | 0.536 (0.366) |
| 32 | 0.629 (0.366) | 0.685 (0.463) |
| 64 | 0.731 (0.537) | 0.918 (0.847) |
| 128 | 0.794 (0.648) | 0.984 (0.957) |
| 256 | 0.825 (0.741) | 0.993 (0.966) |

**synthetic-matmul:256x128x192** (reference: optimum, 57.9 µs; mean over 5 seeds, worst seed in parentheses)

| trials | random | model |
|---|---|---|
| 8 | 0.359 (0.096) | 0.359 (0.096) |
| 16 | 0.526 (0.353) | 0.526 (0.353) |
| 32 | 0.640 (0.417) | 0.706 (0.353) |
| 64 | 0.807 (0.733) | 0.882 (0.828) |
| 128 | 0.824 (0.733) | 0.922 (0.908) |
| 256 | 0.876 (0.750) | 0.980 (0.947) |

**synthetic-matmul:32x512x64** (reference: optimum, 10.3 µs; mean over 5 seeds, worst seed in parentheses)

| trials | random | model |
|---|---|---|
| 8 | 0.491 (0.116) | 0.491 (0.116) |
| 16 | 0.654 (0.386) | 0.654 (0.386) |
| 32 | 0.705 (0.386) | 0.822 (0.632) |
| 64 | 0.724 (0.386) | 0.877 (0.702) |
| 128 | 0.831 (0.755) | 0.956 (0.884) |
| 256 | 0.838 (0.755) | 0.971 (0.884) |

## Real tasks (measured)

**conv2d:7090b8fafa74a832** (reference: best found, 321.6 µs; mean over 3 seeds, worst seed in parentheses)

| trials | random | model |
|---|---|---|
| 8 | 0.819 (0.771) | 0.805 (0.752) |
| 16 | 0.917 (0.896) | 0.895 (0.839) |
| 32 | 0.920 (0.903) | 0.921 (0.907) |
| 64 | 0.950 (0.906) | 0.921 (0.907) |
| 96 | 0.976 (0.962) | 0.921 (0.907) |

**conv2d:21949b671f0fd588** (reference: best found, 228.7 µs; mean over 3 seeds, worst seed in parentheses)

| trials | random | model |
|---|---|---|
| 8 | 0.833 (0.773) | 0.990 (0.986) |
| 16 | 0.881 (0.802) | 0.990 (0.986) |
| 32 | 0.928 (0.918) | 0.993 (0.986) |
| 64 | 0.955 (0.926) | 0.994 (0.986) |
| 96 | 0.975 (0.944) | 0.995 (0.986) |

**matmul:b631c5d3088bb152** (reference: best found, 365.3 µs; mean over 3 seeds, worst seed in parentheses)

| trials | random | model |
|---|---|---|
| 8 | 0.686 (0.685) | 0.679 (0.675) |
| 16 | 0.686 (0.685) | 0.712 (0.675) |
| 32 | 0.694 (0.685) | 0.751 (0.680) |
| 64 | 0.765 (0.687) | 0.791 (0.686) |
| 96 | 0.791 (0.709) | 0.836 (0.690) |

**matmul:9ffa54b7784403b1** (reference: best found, 462.4 µs; mean over 3 seeds, worst seed in parentheses)

| trials | random | model |
|---|---|---|
| 8 | 0.812 (0.756) | 0.809 (0.785) |
| 16 | 0.867 (0.756) | 0.809 (0.785) |
| 32 | 0.867 (0.756) | 0.956 (0.872) |
| 64 | 0.876 (0.784) | 0.965 (0.895) |
| 96 | 0.876 (0.784) | 0.984 (0.955) |
