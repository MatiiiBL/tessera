"""Random sequences of schedule primitives never change what a function computes.

Each example builds a small computation with random sizes, then applies a
random sequence of primitives with random arguments drawn from the current
state of the schedule. A primitive may refuse (``ScheduleError``), and then
the schedule must be unchanged; otherwise the function must still verify.
At the end the interpreter must give the same result as the unscheduled
function, up to the rounding of reordered reductions.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

from tessera import tir
from tessera.tir.schedule import Schedule, ScheduleError

from .workloads import Workload, box_filter, elementwise_chain, matmul_bias_relu, row_softmax, run

WORKLOADS: list[Callable[[st.DataObject], Workload]] = [
    lambda d: matmul_bias_relu(
        d.draw(st.integers(1, 7)), d.draw(st.integers(1, 7)), d.draw(st.integers(1, 6))
    ),
    lambda d: box_filter(d.draw(st.integers(1, 6)), d.draw(st.integers(1, 6)), 2),
    lambda d: row_softmax(d.draw(st.integers(1, 5)), d.draw(st.integers(1, 7))),
    lambda d: elementwise_chain(d.draw(st.integers(1, 6)), d.draw(st.integers(1, 6))),
]

PRIMITIVES = [
    "split",
    "split3",
    "fuse",
    "reorder",
    "parallel",
    "vectorize",
    "unroll",
    "bind",
    "compute_inline",
    "reverse_compute_inline",
    "compute_at",
    "reverse_compute_at",
    "cache_read",
    "cache_write",
    "decompose_reduction",
]


def apply_random(s: Schedule, d: st.DataObject) -> str:
    """Apply one random primitive and return its name."""
    name = d.draw(st.sampled_from(PRIMITIVES))
    block = d.draw(st.sampled_from(s.blocks()))
    loops = s.get_loops(block)
    all_loops = [v for b in s.blocks() for v in s.get_loops(b)]
    if name == "split" and loops:
        loop = d.draw(st.sampled_from(loops))
        s.split(loop, [None, d.draw(st.integers(1, s.extent(loop) + 1))])
    elif name == "split3" and loops:
        loop = d.draw(st.sampled_from(loops))
        s.split(loop, [d.draw(st.integers(1, 3)), None, d.draw(st.integers(1, 3))])
    elif name == "fuse" and len(loops) >= 2:
        k = d.draw(st.integers(0, len(loops) - 2))
        s.fuse(loops[k], loops[k + 1])
    elif name == "reorder" and len(loops) >= 2:
        chosen = d.draw(st.permutations(loops).map(lambda p: p[: max(2, len(p) // 2 + 1)]))
        s.reorder(*chosen)
    elif name in ("parallel", "vectorize", "unroll") and loops:
        getattr(s, name)(d.draw(st.sampled_from(loops)))
    elif name == "bind" and loops:
        tag = d.draw(st.sampled_from(["threadgroup.x", "thread.x", "thread.y"]))
        s.bind(d.draw(st.sampled_from(loops)), tag)
    elif name in ("compute_inline", "reverse_compute_inline"):
        getattr(s, name)(block)
    elif name in ("compute_at", "reverse_compute_at"):
        # Loops of later blocks for compute_at, of earlier ones for reverse_compute_at.
        names = s.blocks()
        k = names.index(block)
        others = names[k + 1 :] if name == "compute_at" else names[:k]
        candidates = [v for b in others for v in s.get_loops(b)] or all_loops
        if candidates:
            getattr(s, name)(block, d.draw(st.sampled_from(candidates)))
    elif name == "cache_read":
        s.cache_read(block, d.draw(st.integers(0, 2)), d.draw(st.sampled_from(["local", "shared"])))
    elif name == "cache_write":
        s.cache_write(block, 0, "local")
    elif name == "decompose_reduction" and loops:
        s.decompose_reduction(block, d.draw(st.sampled_from(loops)))
    return name


@settings(
    max_examples=300,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@given(st.data())
def test_random_schedules_preserve_results(data: st.DataObject) -> None:
    w = data.draw(st.sampled_from(WORKLOADS))(data)
    expected = run(w.func, w)
    s = Schedule(w.func)
    for _ in range(data.draw(st.integers(1, 10))):
        before = s.func
        try:
            name = apply_random(s, data)
        except ScheduleError:
            assert s.func is before
            continue
        event(f"applied {name}" if s.func is not before else "no-op")
        tir.verify(s.func)
    got = run(s.func, w)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5, err_msg="\n".join(s.trace))
