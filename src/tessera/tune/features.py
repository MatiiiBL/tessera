"""Features of a scheduled loop nest, for the cost model.

A scheduled ``PrimFunc`` becomes a fixed-length vector of numbers that
describe how its statements execute: the loops on each statement's path
(extents and kinds), how many bytes each buffer touches inside each loop
level, arithmetic intensity, vector width, parallel extent and the use of
threadgroup and local memory. The cost model never sees the configuration
itself, only these features, so one model can learn from every task.

Each statement (a ``BufferStore`` or an intrinsic ``Evaluate``) gets a
vector of ``STMT_FEATURES`` numbers. The function's vector is the vector of
its heaviest statement (most floating-point operations, then most
iterations), the elementwise maximum over the other statements, and a few
function-wide totals. Counts and sizes are stored as ``log2(1 + x)``.

Bytes touched are estimated per buffer dimension from the index
expressions: an index that is affine in the loop variables (the usual case)
spans ``1 + sum(|coefficient| * (extent - 1))`` over the loops that vary;
other indices (``floordiv``/``floormod`` from fused loops) are bounded by
interval arithmetic. Spans are clipped to the buffer's shape, so the
estimate is exact for dense tiles and an upper bound for strided ones.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import log2, prod

import numpy as np
import numpy.typing as npt

from tessera.tir.expr import (
    AddressOf,
    Binary,
    BufferLoad,
    Call,
    Cast,
    Compare,
    Expr,
    IntImm,
    Logical,
    Not,
    Select,
    Var,
)
from tessera.tir.stmt import (
    Block,
    BlockRealize,
    Buffer,
    BufferStore,
    Evaluate,
    For,
    ForKind,
    IfThenElse,
    LetStmt,
    PrimFunc,
    SeqStmt,
    Stmt,
)

PATH_LEVELS = 6  # innermost loops described one by one
CURVE_LEVELS = 8  # loop levels (from the innermost) of the footprint and intensity curves
BUFFER_SLOTS = 3  # the written buffer, then the two largest reads
LOOP_KINDS = (
    ForKind.SERIAL,
    ForKind.PARALLEL,
    ForKind.VECTORIZED,
    ForKind.UNROLLED,
    ForKind.THREAD_BINDING,
)

_STMT_SCALARS = (
    "iterations",
    "flops_per_iter",
    "flops",
    "loops",
    "vector_width",
    "vector_innermost",
    "unroll",
    "parallel_extent",
    "parallel_depth",
    "threadgroups",
    "threads_per_group",
    "predicated",
    "intrinsic",
    "loads",
    "init",
)
_BUFFER_SCALARS = ("present", "write", "stride", "unit_stride", "reuse")
_GLOBALS = ("statements", "blocks", "shared_bytes", "local_bytes", "total_flops", "total_iters")


def _stmt_names() -> list[str]:
    names: list[str] = []
    for level in range(PATH_LEVELS):
        names.append(f"path{level}.extent")
        names.extend(f"path{level}.{kind.value}" for kind in LOOP_KINDS)
    names.extend(_STMT_SCALARS)
    for slot in range(BUFFER_SLOTS):
        names.extend(f"buf{slot}.{s}" for s in _BUFFER_SCALARS)
        names.extend(f"buf{slot}.bytes{level}" for level in range(CURVE_LEVELS))
    names.extend(f"intensity{level}" for level in range(CURVE_LEVELS))
    return names


STMT_FEATURE_NAMES: tuple[str, ...] = tuple(_stmt_names())
STMT_FEATURES = len(STMT_FEATURE_NAMES)
FEATURE_NAMES: tuple[str, ...] = (
    *(f"main.{n}" for n in STMT_FEATURE_NAMES),
    *(f"other.{n}" for n in STMT_FEATURE_NAMES),
    *(f"func.{n}" for n in _GLOBALS),
)
NUM_FEATURES = len(FEATURE_NAMES)

type FeatureVector = npt.NDArray[np.float64]


def extract(func: PrimFunc) -> FeatureVector:
    """The feature vector of a scheduled function (``NUM_FEATURES`` float64s)."""
    walker = _Walker()
    walker.visit(func.body)
    stmts = walker.statements
    out = np.zeros(NUM_FEATURES)
    if stmts:
        vectors = [_stmt_vector(s) for s in stmts]
        main = max(range(len(stmts)), key=lambda i: (stmts[i].total_flops, stmts[i].iterations))
        out[:STMT_FEATURES] = vectors[main]
        others = [v for i, v in enumerate(vectors) if i != main]
        if others:
            out[STMT_FEATURES : 2 * STMT_FEATURES] = np.max(others, axis=0)
    totals = (
        len(stmts),
        walker.blocks,
        walker.alloc_bytes.get("shared", 0),
        walker.alloc_bytes.get("local", 0),
        sum(s.total_flops for s in stmts),
        sum(s.iterations for s in stmts),
    )
    out[2 * STMT_FEATURES :] = [_lg(totals[0]), _lg(totals[1]), *(_lg(t) for t in totals[2:])]
    return out


def describe(vector: FeatureVector) -> dict[str, float]:
    """The non-zero entries of a feature vector by name (for debugging and docs)."""
    return {name: float(x) for name, x in zip(FEATURE_NAMES, vector, strict=True) if x != 0.0}


# -- walking the function ------------------------------------------------------


@dataclass(frozen=True)
class _Loop:
    var: Var
    min: int
    extent: int
    kind: ForKind
    thread_tag: str | None


@dataclass
class _Access:
    buffer: Buffer
    indices: tuple[Expr, ...]
    write: bool


@dataclass
class _Statement:
    loops: tuple[_Loop, ...]
    accesses: list[_Access]
    subst: dict[Var, Expr]
    flops_per_iter: int
    loads: int
    predicated: bool
    intrinsic: bool
    init: bool
    iterations: int = field(init=False)
    total_flops: int = field(init=False)

    def __post_init__(self) -> None:
        self.iterations = prod(loop.extent for loop in self.loops)
        self.total_flops = self.iterations * self.flops_per_iter


class _Walker:
    def __init__(self) -> None:
        self.statements: list[_Statement] = []
        self.blocks = 0
        self.alloc_bytes: dict[str, int] = {}
        self._loops: list[_Loop] = []
        self._subst: dict[Var, Expr] = {}
        self._predicated = False
        self._init = False

    def visit(self, stmt: Stmt) -> None:
        if isinstance(stmt, For):
            lo = _const(stmt.min, self._subst, default=0)
            extent = max(1, _const(stmt.extent, self._subst, default=1))
            self._loops.append(_Loop(stmt.loop_var, lo, extent, stmt.kind, stmt.thread_tag))
            self.visit(stmt.body)
            self._loops.pop()
        elif isinstance(stmt, BlockRealize):
            self._realize(stmt)
        elif isinstance(stmt, Block):
            self._block(stmt)
        elif isinstance(stmt, SeqStmt):
            for s in stmt.seq:
                self.visit(s)
        elif isinstance(stmt, IfThenElse):
            saved = self._predicated
            self._predicated = True
            self.visit(stmt.then_case)
            if stmt.else_case is not None:
                self.visit(stmt.else_case)
            self._predicated = saved
        elif isinstance(stmt, LetStmt):
            self._subst[stmt.var] = stmt.value
            self.visit(stmt.body)
        elif isinstance(stmt, BufferStore):
            self._store(stmt)
        elif isinstance(stmt, Evaluate):
            self._evaluate(stmt)

    def _realize(self, realize: BlockRealize) -> None:
        for iv, value in zip(realize.block.iter_vars, realize.iter_values, strict=False):
            self._subst[iv.var] = value
        saved = self._predicated
        if not (isinstance(realize.predicate, IntImm) and realize.predicate.value == 1):
            self._predicated = True
        self._block(realize.block)
        self._predicated = saved

    def _block(self, block: Block) -> None:
        self.blocks += 1
        for buf in block.alloc_buffers:
            nbytes = buf.numel * buf.dtype.itemsize
            self.alloc_bytes[buf.scope] = self.alloc_bytes.get(buf.scope, 0) + nbytes
        if block.init is not None:
            saved = self._init
            self._init = True
            self.visit(block.init)
            self._init = saved
        self.visit(block.body)

    def _store(self, store: BufferStore) -> None:
        loads: list[BufferLoad] = []
        flops = _count(store.value, loads)
        accesses = [_Access(ld.buffer, ld.indices, write=False) for ld in loads]
        accesses.append(_Access(store.buffer, store.indices, write=True))
        self._emit(accesses, flops, len(loads), intrinsic=False)

    def _evaluate(self, ev: Evaluate) -> None:
        loads: list[BufferLoad] = []
        _count(ev.value, loads)
        accesses = [_Access(ld.buffer, ld.indices, write=False) for ld in loads]
        if isinstance(ev.value, Call):
            for arg in ev.value.args:
                if isinstance(arg, AddressOf):
                    accesses.append(_Access(arg.buffer, arg.indices, write=False))
        self._emit(accesses, 0, len(loads), intrinsic=True)

    def _emit(self, accesses: list[_Access], flops: int, loads: int, *, intrinsic: bool) -> None:
        self.statements.append(
            _Statement(
                loops=tuple(self._loops),
                accesses=accesses,
                subst=dict(self._subst),
                flops_per_iter=flops,
                loads=loads,
                predicated=self._predicated,
                intrinsic=intrinsic,
                init=self._init,
            )
        )


_FLOP_OPS = frozenset({"add", "sub", "mul", "div", "min", "max", "pow"})


def _count(e: Expr, loads: list[BufferLoad]) -> int:
    """Floating-point operations in ``e``; collects the loads it reads."""
    if isinstance(e, BufferLoad):
        loads.append(e)
        return 0
    if isinstance(e, Binary):
        own = 1 if e.op in _FLOP_OPS and e.dtype.is_float else 0
        return own + _count(e.a, loads) + _count(e.b, loads)
    if isinstance(e, Compare | Logical):
        own = 1 if e.a.dtype.is_float else 0
        return own + _count(e.a, loads) + _count(e.b, loads)
    if isinstance(e, Select):
        return 1 + sum(_count(x, loads) for x in (e.cond, e.true_value, e.false_value))
    if isinstance(e, Call):
        own = 2 if e.name == "fma" else 1
        return own + sum(_count(a, loads) for a in e.args)
    if isinstance(e, Cast | Not):
        inner = e.value if isinstance(e, Cast) else e.a
        return _count(inner, loads)
    return 0


# -- index analysis -------------------------------------------------------------

type _Affine = dict[Var | None, int]


def _affine(e: Expr, subst: Mapping[Var, Expr], memo: dict[int, _Affine | None]) -> _Affine | None:
    """``e`` as constant + sum(coefficient * loop variable), or None if it is not affine."""
    key = id(e)
    if key in memo:
        return memo[key]
    result: _Affine | None
    if isinstance(e, IntImm):
        result = {None: e.value}
    elif isinstance(e, Var):
        bound = subst.get(e)
        result = _affine(bound, subst, memo) if bound is not None else {e: 1}
    elif isinstance(e, Cast):
        result = _affine(e.value, subst, memo) if not e.dtype.is_float else None
    elif isinstance(e, Binary) and e.op in ("add", "sub", "mul"):
        a = _affine(e.a, subst, memo)
        b = _affine(e.b, subst, memo)
        result = None if a is None or b is None else _combine(e.op, a, b)
    else:
        result = None
    memo[key] = result
    return result


def _combine(op: str, a: _Affine, b: _Affine) -> _Affine | None:
    if op == "mul":
        if set(a) <= {None}:
            a, b = b, a
        if not set(b) <= {None}:
            return None
        c = b.get(None, 0)
        return {v: k * c for v, k in a.items()}
    sign = 1 if op == "add" else -1
    out = dict(a)
    for v, k in b.items():
        out[v] = out.get(v, 0) + sign * k
    return out


def _bounds(
    e: Expr, subst: Mapping[Var, Expr], ranges: Mapping[Var, tuple[int, int]]
) -> tuple[int, int]:
    """An interval containing every value of the integer expression ``e``."""
    if isinstance(e, IntImm):
        return e.value, e.value
    if isinstance(e, Var):
        if e in ranges:
            return ranges[e]
        bound = subst.get(e)
        return _bounds(bound, subst, ranges) if bound is not None else (0, 0)
    if isinstance(e, Cast):
        return _bounds(e.value, subst, ranges)
    if isinstance(e, Select):
        a = _bounds(e.true_value, subst, ranges)
        b = _bounds(e.false_value, subst, ranges)
        return min(a[0], b[0]), max(a[1], b[1])
    if isinstance(e, Binary):
        a = _bounds(e.a, subst, ranges)
        b = _bounds(e.b, subst, ranges)
        return _binary_bounds(e.op, a, b)
    return 0, 0


def _binary_bounds(op: str, a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    if op == "add":
        return a[0] + b[0], a[1] + b[1]
    if op == "sub":
        return a[0] - b[1], a[1] - b[0]
    if op == "mul":
        products = [x * y for x in a for y in b]
        return min(products), max(products)
    if op == "min":
        return min(a[0], b[0]), min(a[1], b[1])
    if op == "max":
        return max(a[0], b[0]), max(a[1], b[1])
    if op in ("floordiv", "floormod") and b[0] == b[1] and b[0] > 0:
        c = b[0]
        if op == "floordiv":
            return a[0] // c, a[1] // c
        if a[1] - a[0] + 1 >= c or a[0] // c != a[1] // c:
            return 0, c - 1
        return a[0] % c, a[1] % c
    return a[0], a[1]


def _const(e: Expr, subst: Mapping[Var, Expr], *, default: int) -> int:
    if isinstance(e, IntImm):
        return e.value
    lo, hi = _bounds(e, subst, {})
    return hi if lo == hi else default


# -- per-statement vectors --------------------------------------------------------


@dataclass
class _BufferStats:
    buffer: Buffer
    write: bool
    curve: list[float]  # unique bytes with the innermost k + 1 loops varying
    stride: int  # elements between consecutive iterations of the innermost loop
    accesses: int


def _stmt_vector(s: _Statement) -> npt.NDArray[np.float64]:
    v: list[float] = []
    path = list(reversed(s.loops))[:PATH_LEVELS]
    for level in range(PATH_LEVELS):
        if level < len(path):
            loop = path[level]
            v.append(_lg(loop.extent))
            v.extend(1.0 if loop.kind == kind else 0.0 for kind in LOOP_KINDS)
        else:
            v.extend([0.0] * (1 + len(LOOP_KINDS)))

    vector = [lp.extent for lp in s.loops if lp.kind == ForKind.VECTORIZED]
    parallel = [lp for lp in s.loops if lp.kind == ForKind.PARALLEL]
    unrolled = [lp.extent for lp in s.loops if lp.kind == ForKind.UNROLLED]
    tags = [(lp.thread_tag or "", lp.extent) for lp in s.loops]
    groups = prod(e for t, e in tags if t.startswith("threadgroup"))
    threads = prod(e for t, e in tags if t.startswith("thread."))
    depth = next((i + 1 for i, lp in enumerate(s.loops) if lp.kind == ForKind.PARALLEL), 0)
    v.extend(
        [
            _lg(s.iterations),
            _lg(s.flops_per_iter),
            _lg(s.total_flops),
            float(len(s.loops)),
            _lg(vector[-1]) if vector else 0.0,
            1.0 if s.loops and s.loops[-1].kind == ForKind.VECTORIZED else 0.0,
            _lg(prod(unrolled)) if unrolled else 0.0,
            _lg(prod(lp.extent for lp in parallel)) if parallel else 0.0,
            float(depth),
            _lg(groups) if groups > 1 else 0.0,
            _lg(threads) if threads > 1 else 0.0,
            1.0 if s.predicated else 0.0,
            1.0 if s.intrinsic else 0.0,
            float(s.loads),
            1.0 if s.init else 0.0,
        ]
    )

    stats = _buffer_stats(s)
    writes = [b for b in stats if b.write]
    reads = sorted((b for b in stats if not b.write), key=lambda b: (-b.curve[-1], b.buffer.name))
    slots: list[_BufferStats | None] = [*writes[:1], *reads[: BUFFER_SLOTS - len(writes[:1])]]
    slots += [None] * (BUFFER_SLOTS - len(slots))
    for b in slots:
        if b is None:
            v.extend([0.0] * (len(_BUFFER_SCALARS) + CURVE_LEVELS))
            continue
        unique = b.curve[-1] / b.buffer.dtype.itemsize
        v.extend(
            [
                1.0,
                1.0 if b.write else 0.0,
                _lg(b.stride),
                1.0 if b.stride == 1 else 0.0,
                _lg(b.accesses / max(unique, 1.0)),
            ]
        )
        v.extend(_lg(x) for x in b.curve)

    for level in range(CURVE_LEVELS):
        inner = s.loops[max(0, len(s.loops) - level - 1) :]
        work = s.flops_per_iter * prod(lp.extent for lp in inner)
        traffic = sum(b.curve[level] for b in stats)
        v.append(_lg(work / traffic) if traffic > 0 else 0.0)
    return np.asarray(v, dtype=np.float64)


def _buffer_stats(s: _Statement) -> list[_BufferStats]:
    memo: dict[int, _Affine | None] = {}
    by_buffer: dict[int, _BufferStats] = {}
    innermost = next((lp for lp in reversed(s.loops) if lp.extent > 1), None)
    for acc in s.accesses:
        curve = [_footprint(acc, s, level, memo) for level in range(CURVE_LEVELS)]
        stride = _stride(acc, s, innermost, memo) if innermost is not None else 0
        count = s.iterations
        known = by_buffer.get(id(acc.buffer))
        if known is None:
            by_buffer[id(acc.buffer)] = _BufferStats(acc.buffer, acc.write, curve, stride, count)
        else:
            known.write = known.write or acc.write
            known.curve = [max(a, b) for a, b in zip(known.curve, curve, strict=True)]
            known.stride = max(known.stride, stride)
            known.accesses += count
    return list(by_buffer.values())


def _footprint(acc: _Access, s: _Statement, level: int, memo: dict[int, _Affine | None]) -> float:
    """Bytes of ``acc.buffer`` touched while the innermost ``level + 1`` loops run."""
    varying = s.loops[max(0, len(s.loops) - level - 1) :]
    extents = {lp.var: lp.extent for lp in varying}
    elements = 1
    for dim, index in zip(acc.buffer.shape, acc.indices, strict=False):
        aff = _affine(index, s.subst, memo)
        if aff is not None:
            span = 1 + sum(abs(k) * (extents[v] - 1) for v, k in aff.items() if v in extents)
        else:
            ranges = {lp.var: (lp.min, lp.min) for lp in s.loops}
            ranges.update({lp.var: (lp.min, lp.min + lp.extent - 1) for lp in varying})
            lo, hi = _bounds(index, s.subst, ranges)
            span = hi - lo + 1
        elements *= min(max(span, 1), dim)
    return float(elements * acc.buffer.dtype.itemsize)


def _stride(acc: _Access, s: _Statement, loop: _Loop, memo: dict[int, _Affine | None]) -> int:
    """Elements between the addresses of two consecutive iterations of ``loop``."""
    total = 0
    for step, index in zip(acc.buffer.strides, acc.indices, strict=False):
        aff = _affine(index, s.subst, memo)
        if aff is not None:
            total += aff.get(loop.var, 0) * step
        else:
            ranges = {lp.var: (lp.min, lp.min) for lp in s.loops}
            ranges[loop.var] = (loop.min, loop.min + 1)
            lo, hi = _bounds(index, s.subst, ranges)
            total += (hi - lo) * step
    return abs(total)


def _lg(x: float) -> float:
    return log2(1.0 + x) if x > 0 else 0.0
