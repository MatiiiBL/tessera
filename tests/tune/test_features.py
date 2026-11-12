"""Loop-nest features: hand-checked values and invariants over the synthetic space."""

from __future__ import annotations

import random
from math import log2

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from tessera.dtype import DType
from tessera.tir.expr import AddressOf, Call, FloatImm, IntImm, Var, lt
from tessera.tir.stmt import (
    TRUE,
    Block,
    BlockRealize,
    Buffer,
    BufferRegion,
    BufferStore,
    Evaluate,
    For,
    ForKind,
    IterVar,
    PrimFunc,
    Range,
    Stmt,
)
from tessera.tune.features import (
    BUFFER_SLOTS,
    CURVE_LEVELS,
    FEATURE_NAMES,
    NUM_FEATURES,
    describe,
    extract,
)
from tessera.tune.synthetic import SyntheticMatmul


def lg(x: float) -> float:
    return log2(1 + x)


def wrap(body: Stmt, params: tuple[Buffer, ...], allocs: tuple[Buffer, ...] = ()) -> PrimFunc:
    root = Block("root", (), (), (), body, alloc_buffers=allocs)
    return PrimFunc("f", params, BlockRealize((), TRUE, root))


def add_bias(kind_i: ForKind = ForKind.PARALLEL, kind_j: ForKind = ForKind.VECTORIZED) -> PrimFunc:
    """for i in 8 (parallel): for j in 16 (vectorized): C[i, j] = A[i, j] + B[j]"""
    a, b, c = Buffer("A", (8, 16)), Buffer("B", (16,)), Buffer("C", (8, 16))
    i, j, vi, vj = Var("i"), Var("j"), Var("vi"), Var("vj")
    block = Block(
        "C",
        (IterVar(vi, Range.of(8)), IterVar(vj, Range.of(16))),
        (BufferRegion.full(a), BufferRegion.full(b)),
        (BufferRegion.full(c),),
        BufferStore(c, a[vi, vj] + b[vj], (vi, vj)),
    )
    body: Stmt = BlockRealize((i, j), TRUE, block)
    body = For(j, IntImm(0), IntImm(16), kind_j, body)
    body = For(i, IntImm(0), IntImm(8), kind_i, body)
    return wrap(body, (a, b, c))


def test_names_are_unique_and_match_the_vector() -> None:
    assert len(set(FEATURE_NAMES)) == NUM_FEATURES
    assert extract(add_bias()).shape == (NUM_FEATURES,)


def test_hand_checked_values_of_a_simple_nest() -> None:
    f = describe(extract(add_bias()))
    assert f["main.iterations"] == lg(128)
    assert f["main.flops_per_iter"] == lg(1)
    assert f["main.loops"] == 2
    assert f["main.vector_width"] == lg(16)
    assert f["main.vector_innermost"] == 1
    assert f["main.parallel_extent"] == lg(8)
    assert f["main.parallel_depth"] == 1
    assert f["main.path0.extent"] == lg(16)
    assert f["main.path0.vectorized"] == 1
    assert f["main.path1.parallel"] == 1
    # Slot 0 is the written buffer C, then the reads by size: A (512 B), B (64 B).
    assert f["main.buf0.write"] == 1
    assert f["main.buf0.bytes0"] == lg(16 * 4)
    assert f["main.buf0.bytes1"] == lg(8 * 16 * 4)
    assert f["main.buf1.bytes1"] == lg(8 * 16 * 4)
    assert f["main.buf2.bytes0"] == lg(64)
    assert f["main.buf2.bytes1"] == lg(64)  # B is reused across i
    assert f["main.buf1.unit_stride"] == 1
    assert f["main.buf2.reuse"] == lg(8)
    # Padding of the curves past the outermost loop repeats the whole-nest value.
    assert f[f"main.buf0.bytes{CURVE_LEVELS - 1}"] == lg(512)
    assert f["main.intensity1"] == lg(128 / (512 + 512 + 64))
    assert "other.iterations" not in f  # one statement only


def test_loop_kinds_change_the_features() -> None:
    plain = extract(add_bias(ForKind.SERIAL, ForKind.SERIAL))
    fast = extract(add_bias())
    assert not np.array_equal(plain, fast)
    assert describe(plain).get("main.vector_width", 0.0) == 0.0


def test_non_affine_indices_are_bounded() -> None:
    """A fused loop indexes with floordiv/floormod; spans still come out right."""
    a, c = Buffer("A", (4, 8)), Buffer("C", (4, 8))
    f, vi, vj = Var("f"), Var("vi"), Var("vj")
    block = Block(
        "C",
        (IterVar(vi, Range.of(4)), IterVar(vj, Range.of(8))),
        (BufferRegion.full(a),),
        (BufferRegion.full(c),),
        BufferStore(c, a[vi, vj] * 2.0, (vi, vj)),
    )
    body = For(f, IntImm(0), IntImm(32), ForKind.SERIAL, BlockRealize((f // 8, f % 8), TRUE, block))
    d = describe(extract(wrap(body, (a, c))))
    assert d["main.buf0.bytes0"] == lg(4 * 8 * 4)
    assert d["main.buf1.bytes0"] == lg(4 * 8 * 4)
    assert d["main.buf1.stride"] == lg(1)


def test_threads_shared_memory_predicates_and_intrinsics() -> None:
    a = Buffer("A", (64,))
    tile = Buffer("tile", (16,), scope="shared")
    g, t, v = Var("g"), Var("t"), Var("v")
    block = Block(
        "copy",
        (IterVar(v, Range.of(64)),),
        (BufferRegion.full(a),),
        (BufferRegion.full(tile),),
        BufferStore(tile, a[v], (v % 16,)),
    )
    copy = BlockRealize((g * 16 + t,), lt(g * 16 + t, 60), block)
    call = Evaluate(Call("c.microkernel", (AddressOf(a, (IntImm(0),)),), DType.int32))
    body: Stmt = For(t, IntImm(0), IntImm(16), ForKind.THREAD_BINDING, copy, "thread.x")
    body = For(g, IntImm(0), IntImm(4), ForKind.THREAD_BINDING, body, "threadgroup.x")
    from tessera.tir.stmt import SeqStmt

    d = describe(extract(wrap(SeqStmt((body, call)), (a,), allocs=(tile,))))
    assert d["main.threadgroups"] == lg(4)
    assert d["main.threads_per_group"] == lg(16)
    assert d["main.predicated"] == 1
    assert d["main.path0.thread_binding"] == 1
    assert d["other.intrinsic"] == 1
    assert d["func.shared_bytes"] == lg(64)
    assert d["func.statements"] == lg(2)


def test_reduction_init_is_a_statement() -> None:
    task = SyntheticMatmul(16, 16, 16)
    d = describe(extract(task.schedule(task.space.default())))
    assert d["other.init"] == 1
    assert d["main.flops_per_iter"] == lg(2)
    assert d["main.loads"] == 3


def test_empty_function() -> None:
    func = wrap(BufferStore(Buffer("x", (1,)), FloatImm(1.0), (IntImm(0),)), ())
    vector = extract(func)
    assert vector.shape == (NUM_FEATURES,)
    assert describe(vector)["main.buf0.bytes0"] == lg(4)


TASK = SyntheticMatmul(32, 64, 16)


@settings(max_examples=40, deadline=None)
@given(st.integers(0, TASK.space.size - 1))
def test_footprints_grow_outward_and_fit_in_the_buffers(index: int) -> None:
    d = extract(TASK.schedule(TASK.space.get(index)))
    named = dict(zip(FEATURE_NAMES, d, strict=True))
    for slot in range(BUFFER_SLOTS):
        curve = [named[f"main.buf{slot}.bytes{level}"] for level in range(CURVE_LEVELS)]
        assert curve == sorted(curve)
        assert curve[-1] <= lg(max(32 * 16, 16 * 64, 32 * 64) * 4) + 1e-9
    assert np.all(np.isfinite(d))


def test_features_are_deterministic_and_distinguish_configs() -> None:
    rng = random.Random(0)
    points = [TASK.space.sample(rng) for _ in range(50)]
    vectors = [extract(TASK.schedule(p)) for p in points]
    again = [extract(TASK.schedule(p)) for p in points]
    assert all(np.array_equal(a, b) for a, b in zip(vectors, again, strict=True))
    assert len({v.tobytes() for v in vectors}) == len({str(p) for p in points})
