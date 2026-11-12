# Metal: whole models

Measured 2026-11-12T07:41:55Z on Apple M4 (4 performance cores, 10 cores, 16 GB), Darwin 27.0.0, commit `7ed8435`. Reproduce with `uv run python -m tessera.backend.metal.bench models`.

Each model compiled with `tessera.compile(model, target="metal", tuned=False)` (the templates'
default configurations), run once to compare with the ONNX Runtime outputs stored with the
model, then 100 timed runs after 10 warmup runs (wall-clock, inputs in place). Errors are
of the first output, relative to its largest magnitude; top-1 compares the classifiers' argmax.

| Model | ms, median (p10 to p90) | Kernels | Max abs error | Max rel error | Top-1 | Compile (s) |
|---|---|---|---|---|---|---|
| resnet50 | 16.87 (16.78 to 17.02) | 56 | 3.46e-06 | 7.46e-07 | same | 3.1 |
| mobilenet_v2 | 1.93 (1.89 to 2.00) | 54 | 6.20e-06 | 1.71e-06 | same | 1.9 |
| bert | 12.34 (12.22 to 12.44) | 134 | 1.10e-05 | 1.26e-06 |  | 2.6 |
| gpt2 | 16.73 (16.42 to 16.89) | 111 | 2.98e-04 | 1.11e-06 |  | 2.0 |
