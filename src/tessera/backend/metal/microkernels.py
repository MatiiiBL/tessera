"""SIMD-group matrix microkernels, registered as tensorize intrinsics.

``metal.simdgroup_gemm_{fm}x{fn}`` computes ``C += A @ B`` for one SIMD group:
an ``8 fm x 8 fn`` tile of C held as ``fm x fn`` SIMD-group matrices
(``simdgroup_float8x8``) while the group walks the reduction 8 steps at a
time, loading ``fm`` tiles of A and ``fn`` tiles of B from memory and
multiplying-accumulating every pair. Its description is
``tir.intrin.matmul_desc(8 fm, 8 fn)``, so ``Schedule.tensorize`` replaces the
update block of a decomposed matmul reduction whose inner loops cover one
SIMD group's tile and the whole reduction.

The call is ``metal.simdgroup_gemm(a, b, c, row_a, row_b, row_c, K, fm, fn)``:
pointers to the first element of each tile, the buffer dimension that indexes
the rows of each one, the reduction length and the tile counts. The row
strides are not part of the call: the code generator computes them from the
buffers' shapes when it prints the kernel, after it has compacted them.
"""

from __future__ import annotations

from typing import Any

import numpy.typing as npt

from tessera.backend.metal.codegen import CodegenError, _KernelWriter, register_intrinsic
from tessera.dtype import DType
from tessera.tir import intrin
from tessera.tir.expr import AddressOf, Call, Expr, IntImm, Var
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
    IterKind,
    IterVar,
    PrimFunc,
    Range,
    Stmt,
)

CALL = "metal.simdgroup_gemm"
FRAGMENT_TILES = ((1, 1), (1, 2), (2, 1), (2, 2), (1, 4), (4, 1), (2, 4), (4, 2), (1, 8))


def name(fm: int, fn: int, ranks: tuple[int, int, int] = (2, 2, 2)) -> str:
    """The intrinsic for an ``8 fm x 8 fn`` tile whose A, B and C buffers have
    ``ranks`` dimensions (registered on first use)."""
    suffix = "" if ranks == (2, 2, 2) else "_r" + "".join(str(r) for r in ranks)
    intrinsic = f"{CALL}_{fm}x{fn}{suffix}"
    if intrinsic not in intrin.registered():
        intrin.register(intrinsic, gemm_desc(8 * fm, 8 * fn, ranks), _impl(fm, fn), _reference)
    return intrinsic


def gemm_desc(m: int, n: int, ranks: tuple[int, int, int]) -> PrimFunc:
    """``tir.intrin.matmul_desc(m, n)`` for buffers of ``ranks`` dimensions: the
    leading dimensions are indexed at 0, so a batch index of the matched block is
    an offset."""
    if ranks == (2, 2, 2):
        return intrin.matmul_desc(m, n)
    k_extent = Var("K")
    ra, rb, rc = ranks
    a = Buffer("a", (1,) * (ra - 2) + (m, 1))
    b = Buffer("b", (1,) * (rb - 2) + (1, n))
    c = Buffer("c", (1,) * (rc - 2) + (m, n))
    i, j, k = Var("i"), Var("j"), Var("k")
    vi, vj, vk = Var("vi"), Var("vj"), Var("vk")

    def at(rank: int, *last: Expr) -> tuple[Expr, ...]:
        return (*(IntImm(0) for _ in range(rank - 2)), *last)

    block = Block(
        "gemm",
        (
            IterVar(vi, Range.of(m)),
            IterVar(vj, Range.of(n)),
            IterVar(vk, Range(IntImm(0), k_extent), IterKind.REDUCE),
        ),
        (),
        (),
        BufferStore(c, c[at(rc, vi, vj)] + a[at(ra, vi, vk)] * b[at(rb, vk, vj)], at(rc, vi, vj)),
    )
    body: Stmt = BlockRealize((i, j, k), TRUE, block)
    body = For(k, IntImm(0), k_extent, ForKind.SERIAL, body)
    body = For(j, IntImm(0), IntImm(n), ForKind.SERIAL, body)
    body = For(i, IntImm(0), IntImm(m), ForKind.SERIAL, body)
    root = BlockRealize((), TRUE, Block("root", (), (), (), body))
    return PrimFunc(f"gemm_{m}x{n}_r{ra}{rb}{rc}", (a, b, c), root)


def _row_dim(region: BufferRegion) -> int:
    """The buffer dimension a tile's rows run along: of the region's two dimensions
    longer than one (every tile and the reduction are at least 8 long), the first."""
    spans = [i for i, r in enumerate(region.region) if not _is_one(r.extent)]
    if len(spans) != 2:
        raise CodegenError(f"{region.buffer.name}: a tile must span two dimensions")
    return spans[0]


def _is_one(e: object) -> bool:
    return isinstance(e, IntImm) and e.value == 1


def _impl(fm: int, fn: int) -> intrin.Impl:
    def impl(match: intrin.IntrinMatch) -> Stmt:
        a, b, c = match.pointers
        ra, rb, rc = (_row_dim(r) for r in match.regions)
        k = match.sizes["K"]
        args = (a, b, c, IntImm(ra), IntImm(rb), IntImm(rc), IntImm(k), IntImm(fm), IntImm(fn))
        return Evaluate(Call(CALL, args, DType.int32))

    return impl


def _reference(
    a: npt.NDArray[Any], b: npt.NDArray[Any], c: npt.NDArray[Any], *extents: int
) -> None:
    """What the interpreter runs for a tensorized block, on views of the matched
    regions (which may have leading dimensions of one)."""
    product = a.reshape(-1, a.shape[-1]) @ b.reshape(-1, b.shape[-1])
    c[...] = c + product.reshape(c.shape)


def register() -> None:
    """Register every tile size for 2-D buffers with ``tir.intrin`` (once)."""
    for fm, fn in FRAGMENT_TILES:
        name(fm, fn)


def _stride(buffer: Buffer, dim: int) -> int:
    stride = 1
    for extent in buffer.shape[dim + 1 :]:
        stride *= extent
    return stride


def _emit(writer: _KernelWriter, call: Call) -> str:
    a, b, c, ra, rb, rc, k, fm_arg, fn_arg = call.args
    for arg in (a, b, c):
        if not isinstance(arg, AddressOf):
            raise CodegenError(f"{CALL}: tile arguments must be pointers")
    assert isinstance(a, AddressOf)
    assert isinstance(b, AddressOf)
    assert isinstance(c, AddressOf)
    assert isinstance(ra, IntImm)
    assert isinstance(rb, IntImm)
    assert isinstance(rc, IntImm)
    assert isinstance(k, IntImm)
    assert isinstance(fm_arg, IntImm)
    assert isinstance(fn_arg, IntImm)
    fm, fn = fm_arg.value, fn_arg.value
    lda, ldb, ldc = (
        _stride(a.buffer, ra.value),
        _stride(b.buffer, rb.value),
        _stride(c.buffer, rc.value),
    )
    if k.value % 8:
        raise CodegenError(f"{CALL}: the reduction ({k.value}) is not a multiple of 8")
    pa, pb, pc = (writer.expr(p)[0] for p in (a, b, c))
    t = "simdgroup_float8x8"
    return "\n".join(
        [
            f"// C += A @ B on {8 * fm} x {8 * fn} tiles of one SIMD group, K = {k.value}",
            "{",
            f"  auto a_tile = {pa};",
            f"  auto b_tile = {pb};",
            f"  auto c_tile = {pc};",
            f"  {t} acc[{fm}][{fn}];",
            "  #pragma unroll",
            f"  for (int i = 0; i < {fm}; ++i) {{",
            "    #pragma unroll",
            f"    for (int j = 0; j < {fn}; ++j) {{",
            f"      simdgroup_load(acc[i][j], c_tile + i * {8 * ldc} + j * 8, {ldc});",
            "    }",
            "  }",
            f"  for (int k = 0; k < {k.value}; k += 8) {{",
            f"    {t} a[{fm}];",
            f"    {t} b[{fn}];",
            "    #pragma unroll",
            f"    for (int i = 0; i < {fm}; ++i) {{",
            f"      simdgroup_load(a[i], a_tile + i * {8 * lda} + k, {lda});",
            "    }",
            "    #pragma unroll",
            f"    for (int j = 0; j < {fn}; ++j) {{",
            f"      simdgroup_load(b[j], b_tile + k * {ldb} + j * 8, {ldb});",
            "    }",
            "    #pragma unroll",
            f"    for (int i = 0; i < {fm}; ++i) {{",
            "      #pragma unroll",
            f"      for (int j = 0; j < {fn}; ++j) {{",
            "        simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);",
            "      }",
            "    }",
            "  }",
            "  #pragma unroll",
            f"  for (int i = 0; i < {fm}; ++i) {{",
            "    #pragma unroll",
            f"    for (int j = 0; j < {fn}; ++j) {{",
            f"      simdgroup_store(acc[i][j], c_tile + i * {8 * ldc} + j * 8, {ldc});",
            "    }",
            "  }",
            "}",
        ]
    )


register_intrinsic(CALL, _emit, writes=(2,), collective=True)
register()
