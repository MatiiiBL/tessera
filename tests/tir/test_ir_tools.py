"""Visitors, substitution, the printer, verification and structural hashing."""

from __future__ import annotations

import pytest

from tessera import te, tir
from tessera.dtype import DType
from tessera.tir.expr import BufferLoad, FloatImm, IntImm, Var, lt
from tessera.tir.functor import (
    StmtMutator,
    StmtVisitor,
    free_vars,
    substitute,
    substitute_stmt,
)
from tessera.tir.stmt import (
    TRUE,
    Block,
    BlockRealize,
    Buffer,
    BufferStore,
    For,
    ForKind,
    IterKind,
    IterVar,
    PrimFunc,
    Range,
    SeqStmt,
)
from tessera.tir.verify import VerifyError

from .workloads import matmul_bias_relu


def copy_kernel(n: int = 8, *, extent: int | None = None) -> PrimFunc:
    """B[i] = A[i] * 2, hand-built (one loop, one block)."""
    a, b = Buffer("A", (n,)), Buffer("B", (n,))
    i, vi = Var("i"), Var("vi")
    store = BufferStore(b, a[vi] * 2.0, (vi,))
    block = Block(
        "copy",
        (IterVar(vi, Range.of(n)),),
        (),
        (),
        store,
    )
    from tessera.tir.analysis import with_regions

    realize = BlockRealize((i,), TRUE, with_regions(block))
    loop = For(i, IntImm(0), IntImm(extent or n), ForKind.SERIAL, realize)
    root = Block("root", (), (), (), loop)
    return PrimFunc("copy", (a, b), BlockRealize((), TRUE, root))


# -- functor -----------------------------------------------------------------------


def test_substitute_replaces_vars_and_buffers() -> None:
    i, j = Var("i"), Var("j")
    a, b = Buffer("A", (4,)), Buffer("B", (4,))
    e = a[i] + a[j] * 2.0
    out = substitute(e, {i: j + 1}, {a: b})
    assert tir.script(out) == "B[j + 1] + B[j] * 2.0"
    assert free_vars(out) == {j}
    assert substitute(e, {}) is e


def test_mutator_keeps_identity_of_unchanged_subtrees() -> None:
    f = matmul_bias_relu(4, 4, 4).func
    assert StmtMutator().visit_stmt(f.body) is f.body
    unrelated = Var("unrelated")
    assert substitute_stmt(f.body, {unrelated: IntImm(0)}) is f.body


def test_visitor_sees_every_load() -> None:
    f = matmul_bias_relu(4, 4, 4).func
    names: list[str] = []

    class Loads(StmtVisitor):
        def visit_buffer_load(self, e: BufferLoad) -> None:
            names.append(e.buffer.name)
            super().visit_buffer_load(e)

    Loads().visit_stmt(f.body)
    assert sorted(set(names)) == ["A", "B", "bias", "matmul"]


# -- printer -----------------------------------------------------------------------


def test_script_of_a_matmul() -> None:
    text = tir.script(matmul_bias_relu(4, 6, 5).func)
    assert text.splitlines()[:3] == [
        "@prim_func",
        "def matmul_bias_relu(A: float32[4, 5], B: float32[5, 6], bias: float32[6], "
        "relu: float32[4, 6]):",
        "    matmul = alloc_buffer(float32[4, 6])",
    ]
    for line in (
        "for k in range(5):",
        "with block('matmul'):",
        "v_k = reduce(5, k)",
        "reads(A[v_i, v_k], B[v_k, v_j])",
        "with init():",
        "matmul[v_i, v_j] = matmul[v_i, v_j] + A[v_i, v_k] * B[v_k, v_j]",
        "relu[v_i_1, v_j_1] = max(matmul[v_i_1, v_j_1] + bias[v_j_1], 0.0)",
    ):
        assert line in text


def test_script_disambiguates_equal_names_and_prints_precedence() -> None:
    i1, i2 = Var("i"), Var("i")
    assert tir.script((i1 + i2) * 3) == "(i + i_1) * 3"
    assert tir.script(i1 - (i2 - 1)) == "i - (i_1 - 1)"
    assert tir.script(i1 // 2 % 3) == "i // 2 % 3"
    assert tir.script(FloatImm(float("-inf"))) == "-inf"
    assert tir.script(IntImm(3, DType.int32)) == "int32(3)"


# -- verify ------------------------------------------------------------------------


def test_verify_accepts_te_functions() -> None:
    tir.verify(matmul_bias_relu(3, 4, 5).func)
    tir.verify(copy_kernel())


def test_verify_rejects_bindings_outside_the_domain() -> None:
    with pytest.raises(VerifyError, match="leave its domain"):
        tir.verify(_shifted())


def _shifted() -> PrimFunc:
    f = copy_kernel()
    loop = f.body.block.body  # type: ignore[attr-defined]
    realize = loop.body
    shifted = BlockRealize((loop.loop_var + 1,), TRUE, realize.block)
    new_loop = For(loop.loop_var, loop.min, loop.extent, loop.kind, shifted)
    root = Block("root", (), (), (), new_loop)
    return PrimFunc(f.name, f.params, BlockRealize((), TRUE, root))


def test_verify_accepts_bindings_guarded_by_a_predicate() -> None:
    f = copy_kernel(extent=10)
    with pytest.raises(VerifyError, match="leave its domain"):
        tir.verify(f)
    loop = f.body.block.body  # type: ignore[attr-defined]
    guarded = BlockRealize(loop.body.iter_values, lt(loop.loop_var, 8), loop.body.block)
    new_loop = For(loop.loop_var, loop.min, loop.extent, loop.kind, guarded)
    tir.verify(
        PrimFunc(f.name, f.params, BlockRealize((), TRUE, Block("root", (), (), (), new_loop)))
    )


def test_verify_rejects_malformed_functions() -> None:
    a, b = Buffer("A", (4,)), Buffer("B", (4,))
    i = Var("i")
    root = lambda body: BlockRealize((), TRUE, Block("root", (), (), (), body))  # noqa: E731

    def bad(body: object, match: str, params: tuple[Buffer, ...] = (a, b)) -> None:
        with pytest.raises(VerifyError, match=match):
            tir.verify(PrimFunc("bad", params, root(body)))  # type: ignore[arg-type]

    loop = lambda body, v=i, n=4: For(v, IntImm(0), IntImm(n), ForKind.SERIAL, body)  # noqa: E731
    bad(BufferStore(b, a[i], (i,)), "never defined")
    bad(loop(BufferStore(b, a[i + 1], (i,))), "can leave")
    bad(loop(BufferStore(b, a[i], (i,))), "appears twice", (a, a))
    bad(loop(BufferStore(Buffer("T", (4,)), a[i], (i,))), "neither a parameter")
    bad(loop(loop(BufferStore(b, a[i], (i,)))), "defined twice")
    bad(loop(BufferStore(b, IntImm(1), (i,))), "store of int64")
    vi = Var("vi")
    inner = Block("x", (IterVar(vi, Range.of(4)),), (), (), BufferStore(b, a[i], (vi,)))
    bad(loop(BlockRealize((i,), TRUE, inner)), "outside its scope")
    init_only = Block(
        "y",
        (IterVar(Var("v"), Range.of(4), IterKind.SPATIAL),),
        (),
        (),
        BufferStore(b, a[0], (IntImm(0),)),
        BufferStore(b, a[0], (IntImm(0),)),
    )
    bad(loop(BlockRealize((i,), TRUE, init_only)), "init but no reduction")
    with pytest.raises(VerifyError, match="root block"):
        tir.verify(PrimFunc("bad", (a, b), loop(BufferStore(b, a[i], (i,)))))


def test_verify_rejects_duplicate_block_names() -> None:
    a, b = Buffer("A", (4,)), Buffer("B", (4,))

    def block(k: int) -> BlockRealize:
        return BlockRealize((), TRUE, Block("x", (), (), (), BufferStore(b, a[k], (IntImm(k),))))

    root = Block("root", (), (), (), SeqStmt((block(0), block(1))))
    with pytest.raises(VerifyError, match="two blocks"):
        tir.verify(PrimFunc("f", (a, b), BlockRealize((), TRUE, root)))


# -- structural hash ---------------------------------------------------------------


def test_structural_hash_ignores_names() -> None:
    def mm(prefix: str) -> PrimFunc:
        a = te.placeholder((4, 4), DType.float32, prefix + "A")
        b = te.placeholder((4, 4), DType.float32, prefix + "B")
        k = te.reduce_axis(4, prefix + "k")
        c = te.compute((4, 4), lambda i, j: te.sum(a[i, k] * b[k, j], axis=k), prefix + "C")
        return te.create_prim_func([a, b, c], prefix + "mm")

    assert tir.structural_hash(mm("x")) == tir.structural_hash(mm("y"))
    assert tir.structural_equal(mm("x"), mm("y"))
    other = matmul_bias_relu(4, 4, 4).func
    assert tir.structural_hash(other) != tir.structural_hash(mm("x"))
    assert tir.structural_hash(matmul_bias_relu(4, 4, 5).func) != tir.structural_hash(other)
