"""PrimFunc → Metal Shading Language.

One PrimFunc becomes one ``kernel`` function. Every thread of the launch runs
the whole function body; a ``THREAD_BINDING`` loop is not a loop on the GPU
but the thread's position on one grid axis:

- ``threadgroup.{x,y,z}`` loops read ``threadgroup_position_in_grid`` and
  their extents give the number of threadgroups;
- ``thread.{x,y,z}`` loops give the threadgroup's shape. Two loops bound to
  the same axis may have different extents: the launch uses the largest and
  the smaller loop is guarded. Threadgroups are launched one-dimensional and
  each axis position is computed from ``thread_index_in_threadgroup`` (x
  fastest); ``results/metal/shape.md`` compares this with threadgroups shaped
  like the thread loops.

Buffers allocated inside the function become ``threadgroup`` arrays
(``shared``), thread-private arrays (``local``) or scratch device buffers
bound after the parameters (``global``). ``threadgroup_barrier`` calls are
inserted where one statement writes a shared buffer that a later one reads,
or reads one that a later one overwrites, including across iterations of an
enclosing loop. Index arithmetic is 32-bit when every buffer and loop extent
fits, which is what makes integer division by constants cheap on the GPU.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from tessera.dtype import DType
from tessera.tir import intrin
from tessera.tir.expr import (
    MATH_INTRINSICS,
    AddressOf,
    Binary,
    BufferLoad,
    Call,
    Cast,
    Compare,
    Expr,
    FloatImm,
    IntImm,
    Logical,
    Not,
    Select,
    Var,
    lt,
)
from tessera.tir.stmt import (
    THREAD_TAGS,
    Block,
    BlockRealize,
    Buffer,
    BufferStore,
    Evaluate,
    For,
    ForKind,
    IfThenElse,
    IterKind,
    LetStmt,
    PrimFunc,
    SeqStmt,
    Stmt,
)

BARRIER = "metal.threadgroup_barrier"
# Flags of a barrier call (bit set): which memory it orders.
MEM_THREADGROUP = 1
MEM_DEVICE = 2

MAX_BUFFER_ARGUMENTS = 31
INT32_MAX = 2**31 - 1


class CodegenError(Exception):
    """The function cannot be expressed as one Metal kernel."""


@dataclass(frozen=True)
class KernelBuffer:
    """A device buffer the kernel binds, in binding order."""

    name: str  # the MSL identifier
    dtype: DType
    shape: tuple[int, ...]
    written: bool

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * self.dtype.itemsize


@dataclass(frozen=True)
class Kernel:
    """One generated kernel and how to launch it."""

    name: str
    source: str  # the kernel function (the module prelude is separate)
    params: tuple[KernelBuffer, ...]  # one per PrimFunc parameter, bound first
    scratch: tuple[KernelBuffer, ...]  # global intermediates, bound after the params
    grid: tuple[int, int, int]  # threadgroups per grid
    threads: tuple[int, int, int]  # threads per threadgroup, as launched (one-dimensional)
    shape: tuple[int, int, int]  # the threadgroup's thread.{x,y,z} extents
    threadgroup_memory: int  # bytes of threadgroup arrays
    index_type: str  # "int" or "long"
    helpers: frozenset[str] = frozenset()  # prelude functions the kernel calls

    @property
    def buffers(self) -> tuple[KernelBuffer, ...]:
        return self.params + self.scratch

    @property
    def num_threads(self) -> int:
        return self.threads[0] * self.threads[1] * self.threads[2]

    @property
    def thread_tags(self) -> dict[str, int]:
        return {f"thread.{a}": e for a, e in zip("xyz", self.shape, strict=True)}


# Helpers the generated code may call; each is emitted once per module, in this order.
HELPERS: dict[str, str] = {
    "floordiv": """\
inline int tessera_floordiv(int a, int b) {
  int q = a / b;
  return (q * b != a && ((a < 0) != (b < 0))) ? q - 1 : q;
}
inline long tessera_floordiv(long a, long b) {
  long q = a / b;
  return (q * b != a && ((a < 0) != (b < 0))) ? q - 1 : q;
}""",
    "floormod": """\
inline int tessera_floormod(int a, int b) {
  int r = a % b;
  return (r != 0 && ((r < 0) != (b < 0))) ? r + b : r;
}
inline long tessera_floormod(long a, long b) {
  long r = a % b;
  return (r != 0 && ((r < 0) != (b < 0))) ? r + b : r;
}""",
    "erf": """\
// Abramowitz and Stegun 7.1.26: absolute error below 1.5e-7.
inline float tessera_erf(float x) {
  float t = 1.0f / (1.0f + 0.3275911f * fabs(x));
  float p = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f
            + t * (-1.453152027f + t * 1.061405429f))));
  float y = 1.0f - p * exp(-x * x);
  return copysign(y, x);
}""",
    "pow": """\
// pow for any sign of the base, like NumPy: negative bases need an integral exponent.
inline float tessera_pow(float x, float y) {
  float r = powr(fabs(x), y);
  if (x >= 0.0f) return r;
  if (floor(y) != y) return NAN;
  return fmod(fabs(y), 2.0f) == 1.0f ? -r : r;
}""",
}

PRELUDE = """\
#include <metal_stdlib>
using namespace metal;
"""


def generate(func: PrimFunc, *, max_threadgroup_memory: int = 32768, flat: bool = True) -> Kernel:
    """Generate the MSL kernel of ``func`` and its launch configuration. ``flat``
    launches one-dimensional threadgroups (see the module docstring); without it
    the threadgroup has the thread loops' shape, for comparison."""
    from tessera.backend.metal.lower import (
        ReductionError,
        compact_buffers,
        lower_cross_thread_reductions,
    )

    try:
        func = lower_cross_thread_reductions(func, launch_extents(func.body))
    except ReductionError as error:
        raise CodegenError(str(error)) from error
    func = compact_buffers(func, intrinsic_extents)
    return _KernelWriter(func, max_threadgroup_memory, flat).write()


def launch_extents(body: Stmt) -> dict[str, int]:
    """The launch extent of each grid axis: the largest loop bound to it."""
    out: dict[str, int] = {}
    for s in _walk_stmt(body):
        if isinstance(s, For) and s.kind is ForKind.THREAD_BINDING and s.thread_tag is not None:
            extent = s.extent.value if isinstance(s.extent, IntImm) else 1
            out[s.thread_tag] = max(out.get(s.thread_tag, 1), extent)
    return out


def module_source(kernels: Sequence[Kernel], *, header: str = "") -> str:
    """One MSL source holding the prelude, the helpers the kernels use, and the kernels."""
    names = [k.name for k in kernels]
    if len(set(names)) != len(names):
        raise CodegenError(f"kernel names are not unique: {names}")
    used = set().union(*(k.helpers for k in kernels)) if kernels else set()
    parts = []
    if header:
        parts.append("".join(f"// {line}\n" for line in header.splitlines()))
    parts.append(PRELUDE)
    parts.extend(text for name, text in HELPERS.items() if name in used)
    parts.extend(k.source for k in kernels)
    return "\n".join(p.rstrip("\n") + "\n" for p in parts)


# -- barrier insertion ---------------------------------------------------------------


def barrier(flags: int) -> Evaluate:
    """A ``threadgroup_barrier`` statement ordering the memory in ``flags``."""
    return Evaluate(Call(BARRIER, (IntImm(flags, DType.int32),), DType.int32))


@dataclass(frozen=True)
class _Access:
    buffer: Buffer
    write: bool


@dataclass(frozen=True)
class _Summary:
    """Synchronized accesses of a statement: before its first barrier (``head``),
    after its last (``tail``), and whether it has a barrier at all."""

    head: frozenset[_Access]
    tail: frozenset[_Access]
    has_barrier: bool

    @staticmethod
    def leaf(accesses: frozenset[_Access]) -> _Summary:
        return _Summary(accesses, accesses, False)


_EMPTY = _Summary(frozenset(), frozenset(), False)


def _conflicts(earlier: Iterable[_Access], later: Iterable[_Access]) -> set[Buffer]:
    """Buffers on which ``later`` must wait for ``earlier`` (RAW, WAR, WAW)."""
    writes = {a.buffer for a in earlier if a.write}
    reads = {a.buffer for a in earlier if not a.write}
    out: set[Buffer] = set()
    for a in later:
        if a.buffer in writes or (a.write and a.buffer in reads):
            out.add(a.buffer)
    return out


def _flags(buffers: Iterable[Buffer]) -> int:
    flags = 0
    for b in buffers:
        flags |= MEM_THREADGROUP if b.scope == "shared" else MEM_DEVICE
    return flags


class _BarrierPass:
    """Inserts barriers for the buffers in ``synced`` (shared, plus global scratch
    when threads of one threadgroup share it)."""

    def __init__(self, synced: set[Buffer], writes_of: Callable[[Call], Sequence[int]]) -> None:
        self.synced = synced
        self.writes_of = writes_of

    def run(self, stmt: Stmt) -> Stmt:
        return self.visit(stmt)[0]

    def visit(self, stmt: Stmt) -> tuple[Stmt, _Summary]:
        if isinstance(stmt, SeqStmt):
            return self.visit_seq(list(stmt.seq))
        if isinstance(stmt, For):
            body, summary = self.visit(stmt.body)
            repeats = not (isinstance(stmt.extent, IntImm) and stmt.extent.value <= 1)
            if repeats and stmt.kind is not ForKind.THREAD_BINDING and summary.has_barrier:
                carried = _conflicts(summary.tail, summary.head)
                if carried:
                    body = _seq_of(body, barrier(_flags(carried)))
                    summary = _Summary(summary.head, frozenset(), True)
            return _replace_for(stmt, body), summary
        if isinstance(stmt, BlockRealize):
            block = stmt.block
            init, init_summary = (
                self.visit(block.init) if block.init is not None else (None, _EMPTY)
            )
            body, body_summary = self.visit(block.body)
            summary = _combine([init_summary, body_summary])
            new_block = Block(
                block.name,
                block.iter_vars,
                block.reads,
                block.writes,
                body,
                init,
                block.alloc_buffers,
                block.annotations,
            )
            access = self.loads(stmt.predicate) | frozenset().union(
                *(self.loads(v) for v in stmt.iter_values)
            )
            summary = _combine([_Summary.leaf(access), summary])
            return BlockRealize(stmt.iter_values, stmt.predicate, new_block), summary
        if isinstance(stmt, IfThenElse):
            then, then_summary = self.visit(stmt.then_case)
            other, other_summary = (
                self.visit(stmt.else_case) if stmt.else_case is not None else (None, _EMPTY)
            )
            cond = _Summary.leaf(self.loads(stmt.condition))
            branches = _Summary(
                then_summary.head | other_summary.head,
                then_summary.tail | other_summary.tail,
                then_summary.has_barrier or other_summary.has_barrier,
            )
            return IfThenElse(stmt.condition, then, other), _combine([cond, branches])
        if isinstance(stmt, LetStmt):
            body, summary = self.visit(stmt.body)
            return LetStmt(stmt.var, stmt.value, body), _combine(
                [_Summary.leaf(self.loads(stmt.value)), summary]
            )
        if isinstance(stmt, BufferStore):
            access = self.loads(stmt.value) | frozenset().union(
                *(self.loads(i) for i in stmt.indices)
            )
            if stmt.buffer in self.synced:
                access |= {_Access(stmt.buffer, True)}
            return stmt, _Summary.leaf(access)
        if isinstance(stmt, Evaluate):
            if _is_barrier(stmt):
                return stmt, _Summary(frozenset(), frozenset(), True)
            return stmt, _Summary.leaf(self.call_accesses(stmt.value))
        raise CodegenError(f"unsupported statement {type(stmt).__name__}")

    def visit_seq(self, stmts: list[Stmt]) -> tuple[Stmt, _Summary]:
        out: list[Stmt] = []
        summaries: list[_Summary] = []
        pending: set[_Access] = set()
        for stmt in stmts:
            new, summary = self.visit(stmt)
            conflict = _conflicts(pending, summary.head)
            if conflict:
                out.append(barrier(_flags(conflict)))
                summaries.append(_Summary(frozenset(), frozenset(), True))
                pending = set()
            out.append(new)
            summaries.append(summary)
            pending = set(summary.tail) if summary.has_barrier else pending | summary.tail
        return _seq_of(*out), _combine(summaries)

    def loads(self, expr: Expr) -> frozenset[_Access]:
        return frozenset(
            _Access(e.buffer, False)
            for e in _walk_expr(expr)
            if isinstance(e, BufferLoad | AddressOf) and e.buffer in self.synced
        )

    def call_accesses(self, expr: Expr) -> frozenset[_Access]:
        out = set(self.loads(expr))
        if isinstance(expr, Call):
            for i in self.writes_of(expr):
                arg = expr.args[i]
                if isinstance(arg, AddressOf) and arg.buffer in self.synced:
                    out.add(_Access(arg.buffer, True))
        return frozenset(out)


def _combine(parts: Sequence[_Summary]) -> _Summary:
    head: set[_Access] = set()
    for p in parts:
        head |= p.head
        if p.has_barrier:
            break
    tail: set[_Access] = set()
    for p in reversed(parts):
        tail |= p.tail
        if p.has_barrier:
            break
    return _Summary(frozenset(head), frozenset(tail), any(p.has_barrier for p in parts))


def _is_barrier(stmt: Stmt) -> bool:
    if not isinstance(stmt, Evaluate) or not isinstance(stmt.value, Call):
        return False
    intrinsic = INTRINSICS.get(stmt.value.name)
    return intrinsic is not None and intrinsic.barrier is not None and intrinsic.barrier(stmt.value)


def _seq_of(*stmts: Stmt) -> Stmt:
    flat: list[Stmt] = []
    for s in stmts:
        flat.extend(s.seq if isinstance(s, SeqStmt) else (s,))
    return flat[0] if len(flat) == 1 else SeqStmt(tuple(flat))


def _replace_for(loop: For, body: Stmt) -> For:
    return For(
        loop.loop_var,
        loop.min,
        loop.extent,
        loop.kind,
        body,
        loop.thread_tag,
        loop.annotations,
    )


def insert_barriers(func: PrimFunc, synced: Iterable[Buffer] | None = None) -> PrimFunc:
    """``func`` with a barrier wherever threads must wait for each other's
    accesses to ``synced`` buffers (default: every shared buffer)."""
    if synced is None:
        synced = [b for b in _allocated(func.body) if b.scope == "shared"]
    body = _BarrierPass(set(synced), intrinsic_writes).run(func.body)
    return PrimFunc(func.name, func.params, body, func.attrs)


# -- backend intrinsics ----------------------------------------------------------------


@dataclass(frozen=True)
class Intrinsic:
    """A ``metal.*`` call: how to print it, which pointer arguments it writes, and
    how many elements each pointer argument covers in its trailing dimensions."""

    emit: Callable[[_KernelWriter, Call], str]
    writes: tuple[int, ...] = ()
    statement: bool = True  # used inside Evaluate (False: an expression)
    extents: tuple[tuple[int, tuple[int, ...]], ...] = ()  # (argument, extents)
    collective: bool = False  # every thread of the group must reach it
    barrier: Callable[[Call], bool] | None = None  # whether a call orders all memory


INTRINSICS: dict[str, Intrinsic] = {}


def register_intrinsic(
    name: str,
    emit: Callable[[_KernelWriter, Call], str],
    *,
    writes: Sequence[int] = (),
    statement: bool = True,
    extents: Mapping[int, tuple[int, ...]] | None = None,
    collective: bool = False,
    barrier: Callable[[Call], bool] | None = None,
) -> None:
    """Teach the code generator a ``metal.<name>`` call. A ``collective`` call is
    made by every thread together (SIMD-group functions, barriers), so it cannot
    sit in a branch that depends on the thread; ``barrier`` says whether a call
    also orders every thread's memory accesses, like ``threadgroup_barrier``."""
    covered = tuple(sorted((extents or {}).items()))
    INTRINSICS[name] = Intrinsic(emit, tuple(writes), statement, covered, collective, barrier)


def intrinsic_extents(call: Call, position: int) -> tuple[int, ...] | None:
    """The trailing extents the pointer at ``position`` covers, if the intrinsic says."""
    intrinsic = INTRINSICS.get(call.name)
    if intrinsic is None:
        return None
    return dict(intrinsic.extents).get(position)


def intrinsic_writes(call: Call) -> Sequence[int]:
    intrinsic = INTRINSICS.get(call.name)
    if intrinsic is not None:
        return intrinsic.writes
    # An unknown call may write through any pointer it is given.
    return [i for i, a in enumerate(call.args) if isinstance(a, AddressOf)]


def _emit_barrier(writer: _KernelWriter, call: Call) -> str:
    flags = call.args[0].value if isinstance(call.args[0], IntImm) else MEM_THREADGROUP
    names = []
    if flags & MEM_DEVICE:
        names.append("mem_flags::mem_device")
    if flags & MEM_THREADGROUP or not names:
        names.append("mem_flags::mem_threadgroup")
    return f"threadgroup_barrier({' | '.join(names)});"


register_intrinsic(BARRIER, _emit_barrier, collective=True, barrier=lambda call: True)


def _emit_allreduce(writer: _KernelWriter, call: Call) -> str:
    from tessera.backend.metal.lower import REDUCE_OPS, SIMD_WIDTH

    partial, scratch, code, width_arg = call.args
    assert isinstance(partial, AddressOf)
    assert isinstance(code, IntImm)
    assert isinstance(width_arg, IntImm)
    op, width = REDUCE_OPS[code.value], width_arg.value
    v = writer.access(partial.buffer, partial.indices)

    def combine(a: str, b: str) -> str:
        return f"{a} + {b}" if op == "add" else f"{op}({a}, {b})"

    simd = {"add": "simd_sum", "max": "simd_max", "min": "simd_min"}[op]
    lines = [f"// {op} across the {width} threads of thread.x"]
    if width >= SIMD_WIDTH:
        lines.append(f"{v} = {simd}({v});")
    elif width > 1:
        lines += [
            f"for (ushort offset = {width // 2}; offset > 0; offset >>= 1) {{",
            f"  {v} = {combine(v, f'simd_shuffle_xor({v}, offset)')};",
            "}",
        ]
    if width > SIMD_WIDTH:
        assert isinstance(scratch, AddressOf)
        groups = width // SIMD_WIDTH
        red = writer.namer(scratch.buffer)
        _, y, z = writer.shape
        # Positions on the thread axes; a row is the threads that share y and z.
        tx = writer.namer(writer.axis_vars["thread.x"])
        ty = writer.namer(writer.axis_vars["thread.y"]) if y > 1 else None
        tz = writer.namer(writer.axis_vars["thread.z"]) if z > 1 else None
        rows = [f"{ty} * {groups}" if ty else None, f"{tz} * {y * groups}" if tz else None]
        row = " + ".join(r for r in rows if r)
        first = row or "0"
        row = f"{row} + " if row else ""
        group = f"{tx} / {SIMD_WIDTH}"
        lines += [
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            f"if ({tx} % {SIMD_WIDTH} == 0) {{",
            f"  {red}[{row}{group}] = {v};",
            "}",
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            f"{v} = {red}[{first}];",
            f"for (uint group = 1; group < {groups}; ++group) {{",
            f"  {v} = {combine(v, f'{red}[{row}group]')};",
            "}",
        ]
    return "\n".join(lines)


def _allreduce_synchronizes(call: Call) -> bool:
    from tessera.backend.metal.lower import SIMD_WIDTH

    width = call.args[3]
    return isinstance(width, IntImm) and width.value > SIMD_WIDTH


register_intrinsic(
    "metal.allreduce",
    _emit_allreduce,
    writes=(0, 1),
    collective=True,
    barrier=_allreduce_synchronizes,
)


# SIMD-group matrices (simdgroup_matrix): 8 x 8 tiles held by the 32 lanes of a SIMD
# group together. Arguments that are tiles point into a local buffer whose only uses are
# these calls; the code generator declares it as an array of simdgroup_float8x8.
SIMD_LANES = 32
TILE = (8, 8)
FRAGMENT_ARGS = {
    "metal.simdgroup_fill": (0,),
    "metal.simdgroup_load": (0,),
    "metal.simdgroup_store": (0,),
    "metal.simdgroup_mma": (0, 1, 2, 3),
}


def _fragment_buffers(body: Stmt) -> set[Buffer]:
    out: set[Buffer] = set()
    for s in _walk_stmt(body):
        if isinstance(s, Evaluate) and isinstance(s.value, Call):
            for i in FRAGMENT_ARGS.get(s.value.name, ()):
                arg = s.value.args[i]
                if not isinstance(arg, AddressOf) or arg.buffer.scope != "local":
                    raise CodegenError(
                        f"{s.value.name}: argument {i} must point into a local buffer"
                    )
                buffer = arg.buffer
                if buffer.ndim < 2 or buffer.shape[-2] % 8 or buffer.shape[-1] % 8:
                    raise CodegenError(
                        f"{buffer.name} holds SIMD-group matrices, so its last two dimensions "
                        f"must be multiples of 8, not {buffer.shape}"
                    )
                if buffer.dtype not in (DType.float32, DType.float16):
                    raise CodegenError(
                        f"SIMD-group matrices hold float32 or float16, not {buffer.dtype}"
                    )
                out.add(buffer)
    return out


def _fragment_count(buffer: Buffer) -> int:
    return buffer.numel // 64


def _emit_fill(writer: _KernelWriter, call: Call) -> str:
    frag, value = call.args[:2]
    ctype = _BUFFER_TYPES[_fragment_dtype(frag)]
    return (
        f"{writer.fragment(frag)} = make_filled_simdgroup_matrix<{ctype}, 8, 8>"
        f"({writer.expr(value)[0]});"
    )


def _fragment_dtype(arg: Expr) -> DType:
    assert isinstance(arg, AddressOf)
    return arg.buffer.dtype


def _emit_load(writer: _KernelWriter, call: Call) -> str:
    frag, src, ld, transpose = call.args[:4]
    text = f"simdgroup_load({writer.fragment(frag)}, {writer.expr(src)[0]}, {writer.expr(ld)[0]}"
    if isinstance(transpose, IntImm) and transpose.value:
        text += ", ulong2(0, 0), true"
    return text + ");"


def _emit_store(writer: _KernelWriter, call: Call) -> str:
    frag, dst, ld = call.args[:3]
    return f"simdgroup_store({writer.fragment(frag)}, {writer.expr(dst)[0]}, {writer.expr(ld)[0]});"


def _emit_mma(writer: _KernelWriter, call: Call) -> str:
    d, a, b, c = (writer.fragment(x) for x in call.args[:4])
    return f"simdgroup_multiply_accumulate({d}, {a}, {b}, {c});"


register_intrinsic(
    "metal.simdgroup_fill", _emit_fill, writes=(0,), extents={0: TILE}, collective=True
)
register_intrinsic(
    "metal.simdgroup_load",
    _emit_load,
    writes=(0,),
    extents={0: TILE, 1: TILE},
    collective=True,
)
register_intrinsic(
    "metal.simdgroup_store",
    _emit_store,
    writes=(1,),
    extents={0: TILE, 1: TILE},
    collective=True,
)
register_intrinsic(
    "metal.simdgroup_mma",
    _emit_mma,
    writes=(0,),
    extents={0: TILE, 1: TILE, 2: TILE, 3: TILE},
    collective=True,
)


# In the IR, each call also carries the row stride of its tile buffers (the last
# arguments), which the interpreter's references need to find a tile in a flat
# buffer. The generated code addresses SIMD-group matrices by tile instead.


def _ld(frag: AddressOf) -> IntImm:
    return IntImm(frag.buffer.shape[-1] if frag.buffer.ndim else 1)


def simdgroup_fill(frag: AddressOf, value: Expr) -> Evaluate:
    """Set the 8 x 8 tile at ``frag`` to ``value``."""
    return Evaluate(Call("metal.simdgroup_fill", (frag, value, _ld(frag)), DType.int32))


def simdgroup_load(
    frag: AddressOf, src: AddressOf, ld: int, *, transpose: bool = False
) -> Evaluate:
    """Load the 8 x 8 tile at ``src`` (rows ``ld`` elements apart) into ``frag``."""
    args = (frag, src, IntImm(ld), IntImm(int(transpose)), _ld(frag))
    return Evaluate(Call("metal.simdgroup_load", args, DType.int32))


def simdgroup_store(frag: AddressOf, dst: AddressOf, ld: int) -> Evaluate:
    """Store the tile ``frag`` at ``dst`` (rows ``ld`` elements apart)."""
    args = (frag, dst, IntImm(ld), _ld(frag))
    return Evaluate(Call("metal.simdgroup_store", args, DType.int32))


def simdgroup_mma(d: AddressOf, a: AddressOf, b: AddressOf, c: AddressOf) -> Evaluate:
    """``d = a @ b + c`` on 8 x 8 tiles."""
    args = (d, a, b, c, _ld(d), _ld(a), _ld(b), _ld(c))
    return Evaluate(Call("metal.simdgroup_mma", args, DType.int32))


def _tile(pointer: intrin.Pointer, ld: int) -> npt.NDArray[Any]:
    return pointer.view((8, 8), (ld, 1))


def _fill_reference(frag: intrin.Pointer, value: float, ld: int) -> None:
    _tile(frag, ld)[...] = value


def _load_reference(
    frag: intrin.Pointer, src: intrin.Pointer, ld: int, transpose: int, frag_ld: int
) -> None:
    tile = _tile(src, ld)
    _tile(frag, frag_ld)[...] = tile.T if transpose else tile


def _store_reference(frag: intrin.Pointer, dst: intrin.Pointer, ld: int, frag_ld: int) -> None:
    _tile(dst, ld)[...] = _tile(frag, frag_ld)


def _mma_reference(
    d: intrin.Pointer,
    a: intrin.Pointer,
    b: intrin.Pointer,
    c: intrin.Pointer,
    ld_d: int,
    ld_a: int,
    ld_b: int,
    ld_c: int,
) -> None:
    product = _tile(a, ld_a) @ _tile(b, ld_b) + _tile(c, ld_c)
    _tile(d, ld_d)[...] = product


# The interpreter runs these calls through their references.
for _name, _reference in (
    ("metal.simdgroup_fill", _fill_reference),
    ("metal.simdgroup_load", _load_reference),
    ("metal.simdgroup_store", _store_reference),
    ("metal.simdgroup_mma", _mma_reference),
):
    if _name not in intrin.registered():
        intrin.register(_name, reference=_reference)


# -- expressions ----------------------------------------------------------------------

# C++ precedence levels (smaller binds tighter).
_PRIMARY, _UNARY, _MUL, _ADD, _REL, _EQ, _AND, _OR, _COND = 0, 2, 3, 4, 6, 7, 11, 12, 13

_BINARY_TOKENS = {"add": ("+", _ADD), "sub": ("-", _ADD), "mul": ("*", _MUL), "div": ("/", _MUL)}
_COMPARE_TOKENS = {
    "lt": ("<", _REL),
    "le": ("<=", _REL),
    "gt": (">", _REL),
    "ge": (">=", _REL),
    "eq": ("==", _EQ),
    "ne": ("!=", _EQ),
}
_MATH_FUNCTIONS = {
    "exp": "exp",
    "log": "log",
    "sqrt": "sqrt",
    "rsqrt": "rsqrt",
    "floor": "floor",
    "ceil": "ceil",
    "fma": "fma",
}

_BUFFER_TYPES = {
    DType.float32: "float",
    DType.float16: "half",
    DType.int64: "long",
    DType.int32: "int",
    DType.int8: "char",
    DType.uint8: "uchar",
    DType.bool_: "bool",
}

# Identifiers generated code must not shadow: MSL and C++ keywords, the Metal
# functions the printer calls, and the names it declares itself.
_RESERVED_WORDS = """
    alignas alignof and asm auto bool break case catch char class const constexpr
    const_cast continue decltype default delete do double dynamic_cast else enum
    explicit export extern false float for friend goto if inline int long mutable
    namespace new noexcept not nullptr operator or private protected public register
    reinterpret_cast return short signed sizeof static static_assert static_cast struct
    switch template this thread_local throw true try typedef typeid typename union
    unsigned using virtual void volatile while xor
    kernel vertex fragment device constant threadgroup threadgroup_imageblock thread
    ray_data object_data half uchar ushort uint ulong size_t ptrdiff_t metal
    exp log sqrt rsqrt tanh erf abs fabs floor ceil fma min max pow powr fmod copysign
    clamp select precise fast simd_sum simd_max simd_min simd_shuffle_down
    simdgroup_barrier threadgroup_barrier mem_flags INFINITY NAN
    threadgroup_pos thread_index thread_pos simd_lane simd_group group offset
    group_x group_y group_z thread_x thread_y thread_z
"""
RESERVED = frozenset(_RESERVED_WORDS.split())


def _sanitize(hint: str) -> str:
    name = re.sub(r"\W", "_", hint) or "v"
    if name[0].isdigit():
        name = f"v_{name}"
    if name in RESERVED or name.startswith("tessera_"):
        name = f"{name}_"
    return name


class _Namer:
    def __init__(self) -> None:
        self.used: set[str] = set()
        self.names: dict[object, str] = {}

    def claim(self, obj: Var | Buffer, name: str) -> None:
        """Give ``obj`` a reserved name of the generator's own (the grid positions)."""
        self.used.add(name)
        self.names[obj] = name

    def __call__(self, obj: Var | Buffer, hint: str | None = None) -> str:
        if obj in self.names:
            return self.names[obj]
        base = _sanitize(hint or obj.name)
        name, n = base, 0
        while name in self.used:
            n += 1
            name = f"{base}_{n}"
        self.used.add(name)
        self.names[obj] = name
        return name


def _walk_expr(expr: Expr) -> Iterator[Expr]:
    stack = [expr]
    while stack:
        e = stack.pop()
        yield e
        match e:
            case Binary(_, a, b) | Compare(_, a, b) | Logical(_, a, b):
                stack.extend((b, a))
            case Not(a):
                stack.append(a)
            case Select(c, t, f):
                stack.extend((f, t, c))
            case Cast(_, v):
                stack.append(v)
            case Call(_, args, _):
                stack.extend(reversed(args))
            case BufferLoad(_, indices) | AddressOf(_, indices):
                stack.extend(reversed(indices))
            case _:
                pass


def _walk_stmt(stmt: Stmt) -> Iterator[Stmt]:
    stack = [stmt]
    while stack:
        s = stack.pop()
        yield s
        match s:
            case SeqStmt(seq):
                stack.extend(reversed(seq))
            case For():
                stack.append(s.body)
            case BlockRealize(_, _, block):
                stack.append(block.body)
                if block.init is not None:
                    stack.append(block.init)
            case IfThenElse(_, then, other):
                if other is not None:
                    stack.append(other)
                stack.append(then)
            case LetStmt(_, _, body):
                stack.append(body)
            case _:
                pass


def _stmt_exprs(stmt: Stmt) -> Iterator[Expr]:
    """The expressions directly inside ``stmt`` (not those of nested statements)."""
    match stmt:
        case For():
            yield stmt.min
            yield stmt.extent
        case BlockRealize(values, predicate, _):
            yield from values
            yield predicate
        case BufferStore(_, value, indices):
            yield value
            yield from indices
        case IfThenElse(cond, _, _):
            yield cond
        case LetStmt(_, value, _):
            yield value
        case Evaluate(value):
            yield value
        case _:
            pass


def _allocated(stmt: Stmt) -> list[Buffer]:
    out: list[Buffer] = []
    for s in _walk_stmt(stmt):
        if isinstance(s, BlockRealize):
            out.extend(s.block.alloc_buffers)
    return out


def fold(expr: Expr) -> Expr:
    """Fold integer constants and identities (``x * 1``, ``x + 0``) for readable indices."""
    match expr:
        case Binary(op, a, b):
            a, b = fold(a), fold(b)
            if expr.dtype.is_float:
                return Binary(op, a, b)
            if isinstance(a, IntImm) and isinstance(b, IntImm):
                value = _int_op(op, a.value, b.value)
                if value is not None:
                    return IntImm(value, a.type)
            if op == "add":
                if _is_int(a, 0):
                    return b
                if _is_int(b, 0):
                    return a
                # (x / c) * c + x % c is x: what flattening a (row, column) split gives.
                if (
                    isinstance(a, Binary)
                    and a.op == "mul"
                    and isinstance(a.a, Binary)
                    and a.a.op == "floordiv"
                    and isinstance(b, Binary)
                    and b.op == "floormod"
                    and a.a.a == b.a
                    and a.a.b == b.b == a.b
                ):
                    return b.a
            if op == "sub" and _is_int(b, 0):
                return a
            if op == "mul":
                if _is_int(a, 1):
                    return b
                if _is_int(b, 1):
                    return a
                if _is_int(a, 0) or _is_int(b, 0):
                    return IntImm(0, expr.dtype)
            if op in ("floordiv", "div") and _is_int(b, 1):
                return a
            if (
                op == "floordiv"
                and isinstance(b, IntImm)
                and b.value > 0
                and isinstance(a, Binary)
                and a.op == "mul"
                and isinstance(a.b, IntImm)
                and a.b.value % b.value == 0
            ):
                return fold(a.a * (a.b.value // b.value))
            if op == "floormod" and _is_int(b, 1):
                return IntImm(0, expr.dtype)
            return Binary(op, a, b)
        case Compare(op, a, b):
            return Compare(op, fold(a), fold(b))
        case Logical(op, a, b):
            return Logical(op, fold(a), fold(b))
        case Not(a):
            return Not(fold(a))
        case Select(c, t, f):
            return Select(fold(c), fold(t), fold(f))
        case Cast(t, v):
            return Cast(t, fold(v))
        case Call(name, args, t):
            return Call(name, tuple(fold(a) for a in args), t)
        case BufferLoad(buf, indices):
            return BufferLoad(buf, tuple(fold(i) for i in indices))
        case AddressOf(buf, indices):
            return AddressOf(buf, tuple(fold(i) for i in indices))
        case _:
            return expr


def _is_int(e: Expr, value: int) -> bool:
    return isinstance(e, IntImm) and e.value == value and not e.dtype.is_float


def _int_op(op: str, a: int, b: int) -> int | None:
    if op == "add":
        return a + b
    if op == "sub":
        return a - b
    if op == "mul":
        return a * b
    if op == "floordiv" and b != 0:
        return a // b
    if op == "floormod" and b != 0:
        return a % b
    if op == "min":
        return min(a, b)
    if op == "max":
        return max(a, b)
    return None


def _float_literal(value: float, dtype: DType) -> tuple[str, int]:
    suffix = "h" if dtype is DType.float16 else "f"
    if math.isnan(value):
        return "NAN", _PRIMARY
    if math.isinf(value):
        return ("INFINITY", _PRIMARY) if value > 0 else ("-INFINITY", _UNARY)
    text = str(np.float16(value) if dtype is DType.float16 else np.float32(value))
    if not any(c in text for c in ".en"):
        text += ".0"
    return text + suffix, (_UNARY if text.startswith("-") else _PRIMARY)


# -- the kernel writer ------------------------------------------------------------------


@dataclass
class _Scope:
    divergent: bool = False  # inside a branch that depends on the thread


@dataclass
class _Analysis:
    params: list[Buffer]
    allocs: list[Buffer]
    bindings: dict[str, int] = field(default_factory=dict)  # tag -> launch extent
    written: set[Buffer] = field(default_factory=set)


class _KernelWriter:
    def __init__(self, func: PrimFunc, max_threadgroup_memory: int, flat: bool = True) -> None:
        self.func = func
        self.max_threadgroup_memory = max_threadgroup_memory
        self.flat = flat
        self.namer = _Namer()
        self.lines: list[str] = []
        self.indent = 1
        self.helpers: set[str] = set()
        self.nonneg: set[Var] = set()
        self.thread_vars: set[Var] = set()  # vars that differ between threads of a group
        self.subst: dict[Var, Expr] = {}
        self.uses: set[str] = set()  # builtin kernel arguments the body reads
        self.bindings: dict[str, int] = {}  # grid axis -> launch extent
        self.fragments: set[Buffer] = set()  # local buffers held as SIMD-group matrices
        self.shape: tuple[int, int, int] = (1, 1, 1)  # thread.{x,y,z} extents
        self.axis_vars: dict[str, Var] = {}  # grid axis -> the variable that reads it
        self.narrow = True

    # -- analysis

    def analyze(self) -> _Analysis:
        func = self.func
        for p in func.params:
            if p.scope != "global":
                raise CodegenError(f"parameter {p.name} has scope {p.scope}")
        analysis = _Analysis(list(func.params), _allocated(func.body))
        extents: list[int] = [b.numel for b in (*func.params, *analysis.allocs)]
        self.check_bindings(func.body, analysis, set())
        self.bindings = analysis.bindings
        for s in _walk_stmt(func.body):
            if isinstance(s, For) and isinstance(s.extent, IntImm):
                extents.append(s.extent.value)
            for e in _stmt_exprs(s):
                extents.extend(abs(x.value) for x in _walk_expr(e) if isinstance(x, IntImm))
            if isinstance(s, BufferStore):
                analysis.written.add(s.buffer)
            if isinstance(s, Evaluate) and isinstance(s.value, Call):
                for i in intrinsic_writes(s.value):
                    arg = s.value.args[i]
                    if isinstance(arg, AddressOf):
                        analysis.written.add(arg.buffer)
        self.narrow = all(x <= INT32_MAX for x in extents)
        return analysis

    def check_bindings(self, stmt: Stmt, analysis: _Analysis, active: set[str]) -> None:
        if isinstance(stmt, For):
            inner = active
            if stmt.kind is ForKind.THREAD_BINDING:
                tag = stmt.thread_tag
                if tag not in THREAD_TAGS:
                    raise CodegenError(f"loop {stmt.loop_var.name} bound to unknown axis {tag!r}")
                if tag in active:
                    raise CodegenError(f"loops bound to {tag} are nested")
                if not isinstance(stmt.extent, IntImm) or not isinstance(stmt.min, IntImm):
                    raise CodegenError(f"loop bound to {tag} needs a constant range")
                extent = stmt.extent.value
                analysis.bindings[tag] = max(analysis.bindings.get(tag, 1), extent)
                if tag not in self.axis_vars:
                    level, axis = tag.split(".")
                    name = f"{'group' if level == 'threadgroup' else 'thread'}_{axis}"
                    self.axis_vars[tag] = Var(name, stmt.loop_var.dtype)
                    self.namer.claim(self.axis_vars[tag], name)
                inner = active | {tag}
            self.check_bindings(stmt.body, analysis, inner)
            return
        for child in _children(stmt):
            self.check_bindings(child, analysis, active)

    # -- output

    def write(self) -> Kernel:
        analysis = self.analyze()
        func = self.func
        threads = _xyz(analysis.bindings, "thread")
        grid = _xyz(analysis.bindings, "threadgroup")
        self.fragments = _fragment_buffers(func.body)
        if self.fragments:
            # SIMD-group matrices are held by the 32 lanes of a SIMD group together:
            # the lanes run on thread.x, and thread.y and thread.z index SIMD groups.
            if threads[0] not in (1, SIMD_LANES):
                raise CodegenError(
                    f"{func.name} uses SIMD-group matrices, so thread.x must be its "
                    f"{SIMD_LANES} lanes, not {threads[0]} threads"
                )
            threads = (SIMD_LANES, threads[1], threads[2])
        num_threads = threads[0] * threads[1] * threads[2]
        if num_threads > 1024:
            raise CodegenError(f"{num_threads} threads per threadgroup (Metal allows 1024)")

        shared = [b for b in analysis.allocs if b.scope == "shared"]
        local = [b for b in analysis.allocs if b.scope == "local"]
        scratch = [b for b in analysis.allocs if b.scope == "global"]
        synced = set(shared)
        if scratch and num_threads > 1:
            if grid != (1, 1, 1):
                names = ", ".join(b.name for b in scratch)
                raise CodegenError(
                    f"intermediate {names} is in global memory, which threads of different "
                    "threadgroups cannot share within one kernel: use local or shared scope"
                )
            synced |= set(scratch)
        if len(func.params) + len(scratch) > MAX_BUFFER_ARGUMENTS:
            raise CodegenError(
                f"{func.name} needs {len(func.params) + len(scratch)} buffers "
                f"(Metal binds at most {MAX_BUFFER_ARGUMENTS})"
            )
        threadgroup_memory = sum(b.numel * b.dtype.itemsize for b in shared)
        if threadgroup_memory > self.max_threadgroup_memory:
            raise CodegenError(
                f"{threadgroup_memory} bytes of threadgroup memory "
                f"(the device has {self.max_threadgroup_memory})"
            )

        body = func.body
        if num_threads > 1 and synced:
            body = _BarrierPass(synced, intrinsic_writes).run(body)

        name = _sanitize(func.name)
        params = [
            KernelBuffer(self.namer(b), b.dtype, b.shape, b in analysis.written)
            for b in func.params
        ]
        scratch_buffers = [KernelBuffer(self.namer(b), b.dtype, b.shape, True) for b in scratch]
        # Declarations: threadgroup arrays must be at kernel scope; thread-private
        # arrays are declared there too, which every thread runs once.
        for b in shared:
            self.line(f"threadgroup {_BUFFER_TYPES[b.dtype]} {self.namer(b)}[{b.numel}];")
        for b in local:
            if b in self.fragments:
                count = _fragment_count(b)
                suffix = "" if count == 1 else f"[{count}]"
                self.line(f"simdgroup_{_BUFFER_TYPES[b.dtype]}8x8 {self.namer(b)}{suffix};")
                continue
            suffix = "" if b.numel == 1 else f"[{b.numel}]"
            self.line(f"{_BUFFER_TYPES[b.dtype]} {self.namer(b)}{suffix};")
        # One name per grid axis; every loop bound to the axis reads it.
        self.shape = threads
        for tag in THREAD_TAGS:
            if tag in self.axis_vars:
                level, axis = tag.split(".")
                var = self.axis_vars[tag]
                self.nonneg.add(var)
                ctype = self.value_type(var.dtype)
                if level == "threadgroup":
                    self.uses.add("threadgroup_pos")
                    position = f"({ctype})threadgroup_pos.{axis}"
                else:
                    self.thread_vars.add(var)
                    position = self.thread_position(axis)
                    position = f"({ctype})" + (
                        position if position.isidentifier() else f"({position})"
                    )
                self.line(f"const {ctype} {self.namer(var)} = {position};")
        self.stmt(body, _Scope())

        signature = [
            f"    {'device' if p.written else 'device const'} {_BUFFER_TYPES[p.dtype]}* "
            f"{p.name} [[buffer({i})]]"
            for i, p in enumerate((*params, *scratch_buffers))
        ]
        if "threadgroup_pos" in self.uses:
            signature.append("    uint3 threadgroup_pos [[threadgroup_position_in_grid]]")
        if "thread_index" in self.uses:
            signature.append("    uint thread_index [[thread_index_in_threadgroup]]")
        if "thread_pos" in self.uses:
            signature.append("    uint3 thread_pos [[thread_position_in_threadgroup]]")
        for builtin, attr in (
            ("simd_lane", "thread_index_in_simdgroup"),
            ("simd_group", "simdgroup_index_in_threadgroup"),
        ):
            if builtin in self.uses:
                signature.append(f"    uint {builtin} [[{attr}]]")
        header = [
            f"// {func.name}: {_dims(grid)} threadgroups of {num_threads} threads"
            + (f" ({_dims(threads)})" if num_threads > max(threads) else ""),
        ]
        if num_threads > 1:
            header.append(f"[[max_total_threads_per_threadgroup({num_threads})]]")
        header.append(f"kernel void {name}(")
        source = "\n".join([*header, ",\n".join(signature) + ") {", *self.lines, "}"])
        return Kernel(
            name=name,
            source=source + "\n",
            params=tuple(params),
            scratch=tuple(scratch_buffers),
            grid=grid,
            threads=(num_threads, 1, 1) if self.flat else threads,
            shape=threads,
            threadgroup_memory=threadgroup_memory,
            index_type=self.index_type,
            helpers=frozenset(self.helpers),
        )

    @property
    def index_type(self) -> str:
        return "int" if self.narrow else "long"

    def thread_position(self, axis: str) -> str:
        """The position on thread axis ``axis`` from the flat thread index."""
        if not self.flat:
            self.uses.add("thread_pos")
            return f"thread_pos.{axis}"
        self.uses.add("thread_index")
        x, y, z = self.shape
        if axis == "x":
            return "thread_index" if x == x * y * z else f"thread_index % {x}"
        if axis == "y":
            inner = "thread_index" if x == 1 else f"thread_index / {x}"
            return inner if z == 1 else f"{inner} % {y}"
        return "thread_index" if x * y == 1 else f"thread_index / {x * y}"

    def line(self, text: str) -> None:
        self.lines.append("  " * self.indent + text)

    def open(self, text: str) -> None:
        self.line(text + " {")
        self.indent += 1

    def close(self) -> None:
        self.indent -= 1
        self.line("}")

    # -- statements

    def stmt(self, stmt: Stmt, scope: _Scope) -> None:
        match stmt:
            case SeqStmt(seq):
                for s in seq:
                    self.stmt(s, scope)
            case For():
                self.for_(stmt, scope)
            case BlockRealize():
                self.block(stmt, scope)
            case BufferStore(buffer, value, indices):
                self.store(buffer, value, indices)
            case IfThenElse(cond, then, other):
                inner = _Scope(scope.divergent or self.thread_dependent(cond))
                self.open(f"if ({self.expr_text(cond)})")
                self.stmt(then, inner)
                if other is not None:
                    self.indent -= 1
                    self.line("} else {")
                    self.indent += 1
                    self.stmt(other, inner)
                self.close()
            case LetStmt(var, value, body):
                if self.is_nonneg(fold(self.subst_expr(value))):
                    self.nonneg.add(var)
                if self.thread_dependent(value):
                    self.thread_vars.add(var)
                ctype = self.value_type(var.dtype)
                self.line(f"const {ctype} {self.namer(var)} = {self.expr_text(value)};")
                self.stmt(body, scope)
            case Evaluate(value):
                self.evaluate(value, scope)
            case _:
                raise CodegenError(f"unsupported statement {type(stmt).__name__}")

    def for_(self, loop: For, scope: _Scope) -> None:
        var = loop.loop_var
        lo = fold(self.subst_expr(loop.min))
        extent = fold(self.subst_expr(loop.extent))
        if self.is_nonneg(lo):
            self.nonneg.add(var)
        ctype = self.value_type(var.dtype)
        if loop.kind is ForKind.THREAD_BINDING:
            assert loop.thread_tag is not None
            assert isinstance(extent, IntImm)
            position = self.axis_vars[loop.thread_tag]
            if loop.thread_tag.startswith("thread."):
                self.thread_vars.add(var)
            if _is_int(lo, 0):
                self.subst[var] = position
            else:
                self.line(f"const {ctype} {self.namer(var)} = {self.expr_text(lo + position)};")
            if extent.value < self.bindings[loop.thread_tag]:
                guard = lt(var, fold(lo + extent))
                self.open(f"if ({self.expr_text(guard)})")
                divergent = scope.divergent or loop.thread_tag.startswith("thread.")
                self.stmt(loop.body, _Scope(divergent))
                self.close()
            else:
                self.stmt(loop.body, scope)
            return
        if _is_int(extent, 1) and not self.thread_dependent(lo):
            # A loop of one iteration is its body with the variable at its start.
            self.subst[var] = lo
            self.stmt(loop.body, scope)
            return
        name = self.namer(var)
        if loop.kind is ForKind.UNROLLED:
            self.line("#pragma unroll")
        end = fold(lo + extent) if not _is_int(lo, 0) else extent
        if self.thread_dependent(lo) or self.thread_dependent(extent):
            scope = _Scope(True)
        self.open(
            f"for ({ctype} {name} = {self.expr_text(lo)}; {name} < {self.expr_text(end)}; ++{name})"
        )
        self.stmt(loop.body, scope)
        self.close()

    def block(self, realize: BlockRealize, scope: _Scope) -> None:
        block = realize.block
        if block.name != "root":
            self.line(f"// {block.name}")
        named: dict[Expr, Expr] = {}
        for iv, value in zip(block.iter_vars, realize.iter_values, strict=True):
            value = fold(self.subst_expr(value))
            if isinstance(value, Var | IntImm):
                self.subst[iv.var] = value
                continue
            named[value] = iv.var
            if self.is_nonneg(value):
                self.nonneg.add(iv.var)
            if self.thread_dependent(value):
                self.thread_vars.add(iv.var)
            ctype = self.value_type(iv.var.dtype)
            self.line(f"const {ctype} {self.namer(iv.var)} = {self.expr_text(value)};")
        # The predicate usually restates an iteration variable's value: use its name.
        predicate = _replace(fold(self.subst_expr(realize.predicate)), named)
        guarded = not _is_int(predicate, 1)
        if guarded:
            scope = _Scope(scope.divergent or self.thread_dependent(predicate))
            self.open(f"if ({self.expr_text(predicate)})")
        if block.init is not None:
            reduce_vars = [iv for iv in block.iter_vars if iv.kind is IterKind.REDUCE]
            conds = [
                self.expr_text(fold(Compare("eq", self.subst_expr(iv.var), iv.dom.min)))
                for iv in reduce_vars
            ]
            if conds:
                self.open(f"if ({' && '.join(conds)})")
                self.stmt(block.init, scope)
                self.close()
            else:
                self.stmt(block.init, scope)
        self.stmt(block.body, scope)
        if guarded:
            self.close()

    def store(self, buffer: Buffer, value: Expr, indices: Sequence[Expr]) -> None:
        target = self.access(buffer, indices)
        value = fold(self.subst_expr(value))
        indices = [fold(self.subst_expr(i)) for i in indices]
        # `C[i] = C[i] + x` reads better as `C[i] += x`.
        if (
            isinstance(value, Binary)
            and value.op == "add"
            and isinstance(value.a, BufferLoad)
            and value.a.buffer is buffer
            and value.a.indices == tuple(indices)
        ):
            self.line(f"{target} += {self.expr_text(value.b, substituted=True)};")
            return
        self.line(f"{target} = {self.expr_text(value, substituted=True)};")

    def evaluate(self, value: Expr, scope: _Scope) -> None:
        if not isinstance(value, Call):
            self.line(f"{self.expr_text(value)};")
            return
        intrinsic = INTRINSICS.get(value.name)
        if intrinsic is not None and intrinsic.collective and scope.divergent:
            what = "a barrier" if value.name == BARRIER else f"{value.name}, which all threads run,"
            raise CodegenError(
                f"{self.func.name}: {what} is needed inside a branch that depends on the "
                "thread (a guarded thread loop, a predicate or a condition)"
            )
        if intrinsic is None or not intrinsic.statement:
            self.line(f"{self.expr_text(value)};")
            return
        call = Call(value.name, tuple(fold(self.subst_expr(a)) for a in value.args), value.type)
        for line in intrinsic.emit(self, call).splitlines():
            self.line(line)

    # -- expressions

    def subst_expr(self, expr: Expr) -> Expr:
        if not self.subst:
            return expr
        return _substitute(expr, self.subst)

    def expr_text(self, expr: Expr, *, substituted: bool = False) -> str:
        """Print ``expr`` (after substituting block variables)."""
        if not substituted:
            expr = fold(self.subst_expr(expr))
        return self.expr(expr)[0]

    def value_type(self, dtype: DType) -> str:
        if dtype is DType.int64 and self.narrow:
            return "int"
        return _BUFFER_TYPES[dtype]

    def is_nonneg(self, expr: Expr) -> bool:
        match expr:
            case IntImm(value, _):
                return value >= 0
            case Var():
                return expr in self.nonneg
            case Binary(op, a, b) if op in ("add", "mul", "floordiv", "floormod", "min", "max"):
                return self.is_nonneg(a) and self.is_nonneg(b)
            case Select(_, t, f):
                return self.is_nonneg(t) and self.is_nonneg(f)
            case Cast(t, v) if t.is_int:
                return self.is_nonneg(v)
            case _:
                return False

    def thread_dependent(self, expr: Expr) -> bool:
        """Whether ``expr`` may differ between threads of one threadgroup."""
        expr = self.subst_expr(expr)
        return any(isinstance(e, Var) and e in self.thread_vars for e in _walk_expr(expr))

    def access(self, buffer: Buffer, indices: Sequence[Expr]) -> str:
        name = self.namer(buffer)
        if buffer in self.fragments:
            raise CodegenError(
                f"{buffer.name} holds SIMD-group matrices: only simdgroup intrinsics can use it"
            )
        if buffer.scope == "local" and buffer.numel == 1:
            return name
        if len(indices) != buffer.ndim:
            raise CodegenError(
                f"{buffer.name} has {buffer.ndim} dimensions, indexed with {len(indices)}"
            )
        flat: Expr | None = None
        for index, stride in zip(indices, buffer.strides, strict=True):
            index = fold(self.subst_expr(index))
            term = fold(index * stride) if stride != 1 else index
            flat = term if flat is None else fold(flat + term)
        if flat is None:
            flat = IntImm(0)
        return f"{name}[{self.expr(flat)[0]}]"

    def fragment(self, arg: Expr) -> str:
        """The SIMD-group matrix holding the 8 x 8 tile that starts at ``arg``."""
        if not isinstance(arg, AddressOf) or arg.buffer not in self.fragments:
            raise CodegenError(f"expected a pointer into a SIMD-group matrix buffer, got {arg}")
        buffer = arg.buffer
        name = self.namer(buffer)
        if _fragment_count(buffer) == 1:
            return name
        # Tiles in row-major order: the last two dimensions count 8 x 8 tiles.
        tiles = (*buffer.shape[:-2], buffer.shape[-2] // 8, buffer.shape[-1] // 8)
        indices = [fold(self.subst_expr(i)) for i in arg.indices]
        indices[-2] = fold(indices[-2] // 8)
        indices[-1] = fold(indices[-1] // 8)
        flat: Expr = IntImm(0)
        stride = 1
        for index, extent in zip(reversed(indices), reversed(tiles), strict=True):
            flat = fold(flat + (fold(index * stride) if stride != 1 else index))
            stride *= extent
        return f"{name}[{self.expr(flat)[0]}]"

    def operand(self, expr: Expr, prec: int, *, right: bool = False) -> str:
        text, p = self.expr(expr)
        if p > prec or (right and p == prec and p != _PRIMARY):
            return f"({text})"
        return text

    def expr(self, e: Expr) -> tuple[str, int]:
        match e:
            case IntImm(value, dtype):
                if dtype is DType.bool_:
                    return ("true" if value else "false"), _PRIMARY
                text = str(value)
                if dtype is DType.int64 and not self.narrow:
                    text += "L"
                return text, (_UNARY if value < 0 else _PRIMARY)
            case FloatImm(value, dtype):
                return _float_literal(value, dtype)
            case Var():
                if e in self.subst:
                    return self.expr(self.subst[e])
                return self.namer(e), _PRIMARY
            case Binary(op, a, b):
                return self.binary(op, a, b, e.dtype)
            case Compare(op, a, b):
                token, prec = _COMPARE_TOKENS[op]
                text = f"{self.operand(a, prec)} {token} {self.operand(b, prec, right=True)}"
                return text, prec
            case Logical(op, a, b):
                token, prec = ("&&", _AND) if op == "and" else ("||", _OR)
                left = self.operand(a, _EQ if isinstance(a, Logical) and a.op != op else prec)
                right = self.operand(
                    b, _EQ if isinstance(b, Logical) and b.op != op else prec, right=True
                )
                return f"{left} {token} {right}", prec
            case Not(a):
                return f"!{self.operand(a, _UNARY)}", _UNARY
            case Select(c, t, f):
                text = f"{self.operand(c, _OR)} ? {self.operand(t, _OR)} : {self.operand(f, _OR)}"
                return text, _COND
            case Cast(dtype, value):
                if dtype is DType.bool_ and value.dtype.is_float:
                    return f"{self.operand(value, _EQ)} != 0.0f", _EQ
                target = self.value_type(dtype)
                if target == self.value_type(value.dtype):
                    return self.expr(value)
                return f"({target}){self.operand(value, _UNARY)}", _UNARY
            case Call():
                return self.call(e), _PRIMARY
            case BufferLoad(buffer, indices):
                text = self.access(buffer, indices)
                if buffer.dtype is DType.int64 and self.narrow:
                    return f"(int){text}", _UNARY
                return text, _PRIMARY
            case AddressOf(buffer, indices):
                if buffer.scope == "local" and buffer.numel == 1:
                    return f"&{self.namer(buffer)}", _UNARY
                return f"&{self.access(buffer, indices)}", _UNARY
            case _:
                raise CodegenError(f"unsupported expression {type(e).__name__}")

    def binary(self, op: str, a: Expr, b: Expr, dtype: DType) -> tuple[str, int]:
        if op in _BINARY_TOKENS:
            token, prec = _BINARY_TOKENS[op]
            return f"{self.operand(a, prec)} {token} {self.operand(b, prec, right=True)}", prec
        if op in ("min", "max"):
            return f"{op}({self.expr(a)[0]}, {self.expr(b)[0]})", _PRIMARY
        if op == "pow":
            return self.power(a, b), _PRIMARY
        if op in ("floordiv", "floormod"):
            if dtype.is_float:
                quotient = f"floor({self.operand(a, _MUL)} / {self.operand(b, _MUL, right=True)})"
                if op == "floordiv":
                    return quotient, _PRIMARY
                text = f"{self.operand(a, _ADD)} - {quotient} * {self.operand(b, _MUL, right=True)}"
                return text, _ADD
            if self.is_nonneg(a) and isinstance(b, IntImm) and b.value > 0:
                token = "/" if op == "floordiv" else "%"
                return f"{self.operand(a, _MUL)} {token} {self.operand(b, _MUL, right=True)}", _MUL
            self.helpers.add(op)
            return f"tessera_{op}({self.expr(a)[0]}, {self.expr(b)[0]})", _PRIMARY
        raise CodegenError(f"unsupported operator {op}")

    def power(self, base: Expr, exponent: Expr) -> str:
        x = self.expr(base)[0]
        if isinstance(exponent, FloatImm):
            if exponent.value == 2.0:
                operand = self.operand(base, _MUL)
                return f"({operand} * {operand})"
            if exponent.value == 0.5:
                return f"sqrt({x})"
            if exponent.value == 1.0:
                return f"({x})"
        self.helpers.add("pow")
        return f"tessera_pow({x}, {self.expr(exponent)[0]})"

    def call(self, call: Call) -> str:
        args = [self.expr(a)[0] for a in call.args]
        name = call.name
        if name in MATH_INTRINSICS:
            if name in _MATH_FUNCTIONS:
                return f"{_MATH_FUNCTIONS[name]}({', '.join(args)})"
            if name == "abs":
                return f"{'fabs' if call.dtype.is_float else 'abs'}({args[0]})"
            if name == "tanh":
                # Fast-math tanh returns NaN for large inputs; tanh(±9) is ±1 in float32.
                return f"tanh(clamp({args[0]}, -9.0f, 9.0f))"
            if name == "sigmoid":
                return f"1.0f / (1.0f + exp(-{self.operand(call.args[0], _UNARY, right=True)}))"
            if name == "erf":
                self.helpers.add("erf")
                return f"tessera_erf({args[0]})"
        intrinsic = INTRINSICS.get(name)
        if intrinsic is not None and not intrinsic.statement:
            return intrinsic.emit(self, call)
        raise CodegenError(f"unknown intrinsic {name}")


def _children(stmt: Stmt) -> list[Stmt]:
    match stmt:
        case SeqStmt(seq):
            return list(seq)
        case For():
            return [stmt.body]
        case BlockRealize(_, _, block):
            return [block.init, block.body] if block.init is not None else [block.body]
        case IfThenElse(_, then, other):
            return [then, other] if other is not None else [then]
        case LetStmt(_, _, body):
            return [body]
        case _:
            return []


def _substitute(expr: Expr, mapping: dict[Var, Expr]) -> Expr:
    match expr:
        case Var():
            return mapping.get(expr, expr)
        case Binary(op, a, b):
            return Binary(op, _substitute(a, mapping), _substitute(b, mapping))
        case Compare(op, a, b):
            return Compare(op, _substitute(a, mapping), _substitute(b, mapping))
        case Logical(op, a, b):
            return Logical(op, _substitute(a, mapping), _substitute(b, mapping))
        case Not(a):
            return Not(_substitute(a, mapping))
        case Select(c, t, f):
            return Select(_substitute(c, mapping), _substitute(t, mapping), _substitute(f, mapping))
        case Cast(t, v):
            return Cast(t, _substitute(v, mapping))
        case Call(name, args, t):
            return Call(name, tuple(_substitute(a, mapping) for a in args), t)
        case BufferLoad(buf, indices):
            return BufferLoad(buf, tuple(_substitute(i, mapping) for i in indices))
        case AddressOf(buf, indices):
            return AddressOf(buf, tuple(_substitute(i, mapping) for i in indices))
        case _:
            return expr


def _replace(expr: Expr, named: dict[Expr, Expr]) -> Expr:
    """``expr`` with every subexpression found in ``named`` replaced by its value."""
    if not named:
        return expr
    if expr in named:
        return named[expr]
    match expr:
        case Binary(op, a, b):
            return Binary(op, _replace(a, named), _replace(b, named))
        case Compare(op, a, b):
            return Compare(op, _replace(a, named), _replace(b, named))
        case Logical(op, a, b):
            return Logical(op, _replace(a, named), _replace(b, named))
        case Not(a):
            return Not(_replace(a, named))
        case _:
            return expr


def _xyz(bindings: dict[str, int], level: str) -> tuple[int, int, int]:
    x, y, z = (bindings.get(f"{level}.{axis}", 1) for axis in "xyz")
    return x, y, z


def _dims(dims: Sequence[int]) -> str:
    return "x".join(str(d) for d in dims)
