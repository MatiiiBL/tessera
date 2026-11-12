"""The benchmark module's pieces (the full studies run from the command line)."""

from __future__ import annotations

import numpy as np

from tessera.backend.c import bench, run_kernel
from tessera.target import Target


def test_scheduled_gemm_and_conv_compute_the_right_thing() -> None:
    target = Target(isa="generic", num_threads=2)
    func = bench.scheduled(bench.gemm_graph(16, 24, 8), target)
    rng = np.random.default_rng(0)
    a = rng.standard_normal((16, 8)).astype(np.float32)
    b = rng.standard_normal((8, 24)).astype(np.float32)
    c = np.zeros((16, 24), np.float32)
    run_kernel(func, [a, b, c], target)
    np.testing.assert_allclose(c, a @ b, rtol=1e-5, atol=1e-5)
    conv = bench.scheduled(bench.conv_graph(8, 4, 8, 3, 2), target)
    assert [p.shape for p in conv.params] == [(1, 8, 8, 4), (3, 3, 4, 8), (8,), (1, 4, 4, 8)]


def test_conv_flops_and_layer_list() -> None:
    assert bench.conv_flops(224, 3, 64, 7, 2) == 2 * 112 * 112 * 64 * 7 * 7 * 3
    assert len({layer[0] for layer in bench.RESNET50}) == len(bench.RESNET50)


def test_markdown_table() -> None:
    rows = [
        bench.Row("gemm", "64x64x64", "neon t1", 1e-4, 5.2),
        bench.Row("gemm", "64x64x64", "numpy t1", 1e-5, 52.4),
    ]
    info = {
        "date": "2026-11-12", "cpu": "Apple M4", "performance_cores": 4, "system": "Darwin",
        "compiler": "clang", "numpy": "2.5", "load_average": 0.5,
    }  # fmt: skip
    text = bench.markdown("GEMM", rows, info, "note")
    assert "| case | neon t1 | numpy t1 |" in text
    assert "| 64x64x64 | 5 | 52 |" in text
