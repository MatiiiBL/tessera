"""Many generated libraries in one process.

A process loads a library for every model and every kernel it runs, and
never unloads one. On macOS each loaded image with thread-local variables
takes one of the process's 512 pthread keys, so generated code must not
declare any: a test run that loaded a few hundred libraries once ran out of
keys and aborted in whatever created the next one. The check runs in a
subprocess, so that a failure cannot abort the test session.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

LIBRARIES = 12  # of each kind: kernels and models

SCRIPT = textwrap.dedent(
    f"""
    import ctypes
    import sys

    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    sys.path.insert(0, {str(Path(__file__).parent)!r})
    from c_funcs import tiled_matmul

    import tessera
    from tessera.backend.c import run_kernel
    from tessera.target import Target

    libc = ctypes.CDLL(None)
    target = Target(isa="generic", num_threads=2)

    def free_keys():
        keys, key = [], ctypes.c_ulong()
        while libc.pthread_key_create(ctypes.byref(key), None) == 0:
            keys.append(key.value)
        for k in keys:
            libc.pthread_key_delete(ctypes.c_ulong(k))
        return len(keys)

    def model(i):
        # y = relu(x @ W + b), with weights of its own for every i.
        w = numpy_helper.from_array(np.full((8, 8), i, np.float32), "w")
        b = numpy_helper.from_array(np.full((8,), -i, np.float32), "b")
        nodes = [
            helper.make_node("MatMul", ["x", "w"], ["h"]),
            helper.make_node("Add", ["h", "b"], ["z"]),
            helper.make_node("Relu", ["z"], ["y"]),
        ]
        graph = helper.make_graph(
            nodes,
            f"keys_{{i}}",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4, 8])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [4, 8])],
            [w, b],
        )
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    before = free_keys()
    for i in range({LIBRARIES}):
        # A kernel library (rows of tiles in parallel) and a model library.
        a, b = np.ones((8, 8), np.float32), np.full((8, 8), i, np.float32)
        c = np.zeros((8, 8), np.float32)
        run_kernel(tiled_matmul(8, 8, 8, tile_m=4, tile_n=8, name=f"keys_{{i}}"), [a, b, c], target)
        assert (c == 8 * i).all()
        y = tessera.compile(model(i), target=target, tuned=False).run(
            {{"x": np.ones((4, 8), np.float32)}}
        )
        assert (next(iter(y.values())) == 7 * i).all()
    print(before, free_keys())
    """
)


def test_loaded_libraries_take_no_pthread_keys() -> None:
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr[-2000:]
    before, after = map(int, result.stdout.split())
    assert after == before, f"{2 * LIBRARIES} libraries took {before - after} pthread keys"
