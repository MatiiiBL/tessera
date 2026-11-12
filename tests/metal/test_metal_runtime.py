"""The Objective-C shim through ctypes: devices, buffers, libraries, programs."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pytest
from metal_funcs import requires_gpu

from tessera.backend.metal import runtime

pytestmark = requires_gpu

COUNTER = """
#include <metal_stdlib>
using namespace metal;
kernel void add_one(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  x[i] = x[i] + 1.0f;
}
kernel void spin(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  float v = x[i];
  for (int k = 0; k < 2000000; ++k) { v = v * 0.999999f + 1e-6f; }
  x[i] = v;
}
"""


def test_device_reports_its_limits() -> None:
    info = runtime.device().info
    assert info.name
    assert info.max_threads_per_threadgroup >= 256
    assert info.max_threadgroup_memory >= 16384
    assert info.has_unified_memory


def test_buffers_are_zeroed_and_shared_with_numpy() -> None:
    buf = runtime.device().buffer(64)
    view = buf.view(np.float32, (4, 4))
    assert not view.any()
    view[1, 2] = 7.0
    assert buf.view(np.float32, (16,))[6] == 7.0
    assert buf.view(np.float32, (2,), offset=24)[0] == 7.0
    with pytest.raises(ValueError, match="exceeds"):
        buf.view(np.float32, (17,))


def test_views_keep_their_buffer_alive() -> None:
    view = runtime.device().buffer(4096).view(np.int32, (1024,))
    gc.collect()
    view[:] = np.arange(1024)
    assert view.sum() == 1024 * 1023 // 2


def test_compiler_errors_are_reported() -> None:
    with pytest.raises(runtime.MetalError, match="undeclared identifier 'nope'"):
        runtime.device().compile("kernel void k(device float* x [[buffer(0)]]) { x[0] = nope; }")
    library = runtime.device().compile(COUNTER)
    with pytest.raises(runtime.MetalError, match="no kernel named missing"):
        library.pipeline("missing")


def test_one_command_buffer_orders_dependent_launches() -> None:
    device = runtime.device()
    pipeline = device.compile(COUNTER).pipeline("add_one")
    assert pipeline.thread_execution_width == 32
    buf = device.buffer(4 * 4096)
    program = device.program()
    for _ in range(50):
        program.add(pipeline, (16, 1, 1), (256, 1, 1), [(buf, 0)], barrier=True)
    assert len(program) == 50
    times = program.run(2)
    assert len(times.wall) == len(times.gpu) == 2
    assert all(0 < g <= w for g, w in zip(times.gpu, times.wall, strict=True))
    np.testing.assert_array_equal(buf.view(np.float32, (4096,)), 100.0)
    program.run(1, serial=True)
    np.testing.assert_array_equal(buf.view(np.float32, (4096,)), 150.0)


def test_launches_bind_buffer_offsets() -> None:
    device = runtime.device()
    pipeline = device.compile(COUNTER).pipeline("add_one")
    buf = device.buffer(4 * 512)
    program = device.program()
    program.add(pipeline, (1, 1, 1), (256, 1, 1), [(buf, 4 * 256)])
    program.run()
    view = buf.view(np.float32, (512,))
    assert not view[:256].any()
    np.testing.assert_array_equal(view[256:], 1.0)


def test_profile_times_every_launch() -> None:
    device = runtime.device()
    library = device.compile(COUNTER)
    buf = device.buffer(4 * 256)
    program = device.program()
    program.add(library.pipeline("add_one"), (1, 1, 1), (256, 1, 1), [(buf, 0)], label="first")
    program.add(library.pipeline("spin"), (1, 1, 1), (256, 1, 1), [(buf, 0)], label="second")
    seconds = program.profile()
    assert program.labels == ["first", "second"]
    assert len(seconds) == 2
    assert all(s > 0 for s in seconds)
    assert seconds[1] > 10 * seconds[0]


def test_too_many_threads_are_rejected() -> None:
    device = runtime.device()
    pipeline = device.compile(COUNTER).pipeline("add_one")
    with pytest.raises(runtime.MetalError, match="exceed"):
        device.program().add(pipeline, (1, 1, 1), (2048, 1, 1), [(device.buffer(16), 0)])


def test_runs_give_up_after_their_timeout() -> None:
    device = runtime.device()
    pipeline = device.compile(COUNTER).pipeline("spin")
    program = device.program()
    program.add(pipeline, (64, 1, 1), (256, 1, 1), [(device.buffer(4 * 64 * 256), 0)])
    with pytest.raises(runtime.MetalError, match="did not finish"):
        program.run(1, timeout=1e-4)
    del program
    gc.collect()


def test_shim_build_is_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TESSERA_CACHE", str(tmp_path))
    first = runtime.build_shim()
    assert first.parent == tmp_path / "metal"
    mtime = first.stat().st_mtime_ns
    assert runtime.build_shim() == first
    assert first.stat().st_mtime_ns == mtime
    assert [p.name for p in first.parent.iterdir()] == [first.name]
