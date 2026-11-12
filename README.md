# Tessera

A tensor compiler for Apple silicon. Tessera imports ONNX models, optimizes
and fuses their graphs, lowers each fused group to a loop IR, schedules it
with autotuned templates, and generates code for the CPU (portable C with
NEON and SME microkernels) and the GPU (Metal).

Work in progress; see [docs/design.md](docs/design.md).
