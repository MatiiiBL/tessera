# Tuning: random against model-guided search

**Provisional:** measured while the machine was busy (load average 5.97 at the start). Re-run on a quiet machine for final numbers.

Measured 2026-11-12T06:27:14Z on Apple M4 (4 performance cores, 10 cores, 16 GB), Darwin 27.0.0, commit `cccf008`. Reproduce with `tessera bench tuning --trials 256 --seeds 5 --real-tasks 2 --real-trials 96 --target c:sme`.


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
| 256 | 0.825 (0.741) | 0.991 (0.957) |

**synthetic-matmul:256x128x192** (reference: optimum, 57.9 µs; mean over 5 seeds, worst seed in parentheses)

| trials | random | model |
|---|---|---|
| 8 | 0.359 (0.096) | 0.359 (0.096) |
| 16 | 0.526 (0.353) | 0.526 (0.353) |
| 32 | 0.640 (0.417) | 0.706 (0.353) |
| 64 | 0.807 (0.733) | 0.882 (0.828) |
| 128 | 0.824 (0.733) | 0.922 (0.908) |
| 256 | 0.876 (0.750) | 0.981 (0.947) |

**synthetic-matmul:32x512x64** (reference: optimum, 10.3 µs; mean over 5 seeds, worst seed in parentheses)

| trials | random | model |
|---|---|---|
| 8 | 0.491 (0.116) | 0.491 (0.116) |
| 16 | 0.654 (0.386) | 0.654 (0.386) |
| 32 | 0.705 (0.386) | 0.822 (0.632) |
| 64 | 0.724 (0.386) | 0.877 (0.702) |
| 128 | 0.831 (0.755) | 0.956 (0.884) |
| 256 | 0.838 (0.755) | 0.984 (0.963) |

## Real tasks (measured)

**conv2d:4fa03a19cbf28f99** (reference: optimum, 1103.3 µs; mean over 5 seeds, worst seed in parentheses)

| trials | random | model |
|---|---|---|
| 8 | 0.664 (0.549) | 0.664 (0.549) |
| 16 | 0.673 (0.549) | 0.673 (0.549) |
| 32 | 0.856 (0.631) | 0.957 (0.785) |
| 64 | 0.903 (0.785) | 1.000 (1.000) |
| 96 | 0.947 (0.866) | 1.000 (1.000) |

**conv2d:e5d625d03531c2b8** (reference: optimum, 478.8 µs; mean over 5 seeds, worst seed in parentheses)

| trials | random | model |
|---|---|---|
| 8 | 0.690 (0.546) | 0.690 (0.546) |
| 16 | 0.727 (0.546) | 0.727 (0.546) |
| 32 | 0.894 (0.777) | 0.981 (0.904) |
| 64 | 0.923 (0.807) | 1.000 (1.000) |
| 96 | 0.962 (0.904) | 1.000 (1.000) |

**matmul:3ccd12885b88dfe5** (reference: optimum, 1885.7 µs; mean over 5 seeds, worst seed in parentheses)

| trials | random | model |
|---|---|---|
| 8 | 0.565 (0.534) | 0.565 (0.534) |
| 16 | 0.580 (0.534) | 0.580 (0.534) |
| 32 | 0.625 (0.534) | 0.873 (0.807) |
| 64 | 0.762 (0.534) | 0.974 (0.957) |
| 96 | 0.883 (0.803) | 1.000 (1.000) |

**matmul:2e05ae1671757c9c** (reference: optimum, 2335.2 µs; mean over 5 seeds, worst seed in parentheses)

| trials | random | model |
|---|---|---|
| 8 | 0.954 (0.954) | 0.954 (0.954) |
| 16 | 0.954 (0.954) | 0.954 (0.954) |
| 32 | 0.954 (0.954) | 1.000 (1.000) |
| 64 | 0.963 (0.954) | 1.000 (1.000) |
| 96 | 0.963 (0.954) | 1.000 (1.000) |
