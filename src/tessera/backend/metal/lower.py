"""Loop-IR rewrites that run before MSL is printed.

**Buffer compaction.** Schedules create cache buffers with the shape of the
whole tensor (``cache_read``, ``cache_write``) and then move the blocks that
use them under tile loops. On the GPU a ``local`` buffer exists once per
thread and a ``shared`` buffer once per threadgroup, so each only needs the
region one thread (or one threadgroup) touches. ``compact_buffers`` finds the
innermost statement around all uses of a buffer and shrinks every dimension to
the elements its uses touch while that statement runs once, with these
variables held fixed:

- the loops around that statement;
- for ``local`` buffers, every thread-bound loop (a thread sees one value of
  each ``threadgroup.*`` and ``thread.*`` axis);
- for ``shared`` buffers, the ``threadgroup.*`` loops (the threads of a
  threadgroup share the buffer, so ``thread.*`` loops range over their
  extents even when they enclose the statement).

Writes of a reduction block count at the outermost loop of the reduction,
because the value they leave is carried from one iteration to the next.
A dimension whose accesses are not affine in the fixed variables, or whose
fixed parts differ between accesses, keeps its full extent.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from tessera.dtype import INDEX, DType
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
    BufferRegion,
    BufferStore,
    Evaluate,
    For,
    ForKind,
    IfThenElse,
    IterKind,
    LetStmt,
    PrimFunc,
    Range,
    SeqStmt,
    Stmt,
)

type Path = tuple[int, ...]
# A linear combination of variables; the constant is under ``None``.
type Affine = dict[Var | None, int]


def affine(expr: Expr, expand: Callable[[Var], Affine | None] | None = None) -> Affine | None:
    """``expr`` as a linear combination of variables plus a constant, or None.
    ``expand`` gives the form of a variable (default: the variable itself)."""
    if isinstance(expr, IntImm):
        return {None: expr.value}
    if isinstance(expr, Var):
        return expand(expr) if expand is not None else {expr: 1}
    if isinstance(expr, Cast) and expr.type.is_int and expr.value.dtype.is_int:
        return affine(expr.value, expand)
    if isinstance(expr, Binary) and expr.op in ("add", "sub", "mul"):
        a = affine(expr.a, expand)
        b = affine(expr.b, expand)
        if a is None or b is None:
            return None
        if expr.op == "mul":
            if set(a) <= {None}:
                return _scale(b, a.get(None, 0))
            if set(b) <= {None}:
                return _scale(a, b.get(None, 0))
            return None
        sign = 1 if expr.op == "add" else -1
        out = dict(a)
        for k, v in b.items():
            out[k] = out.get(k, 0) + sign * v
        return _clean(out)
    return None


def affine_expr(form: Affine, dtype: DType = INDEX) -> Expr:
    """An expression for ``form``: variables in first-appearance order, then the constant."""
    out: Expr | None = None
    for var, coef in form.items():
        if var is None or coef == 0:
            continue
        term: Expr = var if coef == 1 else var * coef
        out = term if out is None else out + term
    const = form.get(None, 0)
    if out is None:
        return IntImm(const, dtype)
    return out + const if const else out


def _scale(form: Affine, factor: int) -> Affine:
    return _clean({k: v * factor for k, v in form.items()})


def _clean(form: Affine) -> Affine:
    return {k: v for k, v in form.items() if v != 0 or k is None}


@dataclass
class _Loop:
    var: Var
    path: Path
    lo: int | None
    hi: int | None  # inclusive
    tag: str | None


# Every loop bound to one grid axis reads the same position, whatever its variable:
# fixed thread-bound variables are compared as (one variable per axis) + start.
type _Axes = dict[str, Var]


def _axis_form(loop: _Loop, axes: _Axes) -> Affine:
    assert loop.tag is not None
    return {axes[loop.tag]: 1, None: loop.lo or 0}


@dataclass
class _Use:
    path: Path
    loops: tuple[_Loop, ...]  # around the access, outermost first
    starts: tuple[Expr, ...]  # with block and let variables substituted
    extents: tuple[int | None, ...]  # elements touched per dimension (None: unknown)


@dataclass
class _Collected:
    uses: dict[Buffer, list[_Use]] = field(default_factory=dict)
    allocated: list[Buffer] = field(default_factory=list)


class _Collector:
    """Records every access of every allocated buffer."""

    def __init__(self, extents_of: Callable[[Call, int], tuple[int, ...] | None]) -> None:
        self.out = _Collected()
        self.bindings: dict[Var, Expr] = {}
        self.extents_of = extents_of
        self.axes: _Axes = {}

    def walk(
        self,
        stmt: Stmt,
        path: Path,
        loops: tuple[_Loop, ...],
        write_path: Path | None,
        regions: dict[Buffer, tuple[Range, ...]],
    ) -> None:
        if isinstance(stmt, For):
            lo = stmt.min.value if isinstance(stmt.min, IntImm) else None
            hi = (
                lo + stmt.extent.value - 1
                if lo is not None and isinstance(stmt.extent, IntImm)
                else None
            )
            tag = stmt.thread_tag if stmt.kind is ForKind.THREAD_BINDING else None
            if tag is not None:
                self.axes.setdefault(tag, Var(tag.replace(".", "_"), stmt.loop_var.dtype))
            self.exprs(stmt, [stmt.min, stmt.extent], path, loops, regions)
            inner = (*loops, _Loop(stmt.loop_var, path, lo, hi, tag))
            self.walk(stmt.body, (*path, 0), inner, write_path, regions)
        elif isinstance(stmt, SeqStmt):
            for i, s in enumerate(stmt.seq):
                self.walk(s, (*path, i), loops, write_path, regions)
        elif isinstance(stmt, BlockRealize):
            block = stmt.block
            self.out.allocated.extend(block.alloc_buffers)
            self.exprs(stmt, [*stmt.iter_values, stmt.predicate], path, loops, regions)
            reduce_vars: set[Var] = set()
            for iv, value in zip(block.iter_vars, stmt.iter_values, strict=True):
                bound = self.subst(value)
                self.bindings[iv.var] = bound
                if iv.kind is IterKind.REDUCE:
                    reduce_vars |= _vars(bound)
            lift = write_path
            for loop in loops:
                if loop.var in reduce_vars:
                    if lift is None or len(loop.path) < len(lift):
                        lift = loop.path
                    break
            block_regions = {r.buffer: r.region for r in (*block.reads, *block.writes)}
            self.walk(block.body, (*path, 0), loops, lift, block_regions)
            if block.init is not None:
                self.walk(block.init, (*path, 1), loops, lift, block_regions)
        elif isinstance(stmt, IfThenElse):
            self.exprs(stmt, [stmt.condition], path, loops, regions)
            self.walk(stmt.then_case, (*path, 0), loops, write_path, regions)
            if stmt.else_case is not None:
                self.walk(stmt.else_case, (*path, 1), loops, write_path, regions)
        elif isinstance(stmt, LetStmt):
            self.exprs(stmt, [stmt.value], path, loops, regions)
            self.bindings[stmt.var] = self.subst(stmt.value)
            self.walk(stmt.body, (*path, 0), loops, write_path, regions)
        elif isinstance(stmt, BufferStore):
            self.exprs(stmt, [stmt.value, *stmt.indices], path, loops, regions)
            target = write_path if write_path is not None else path
            self.point(stmt.buffer, stmt.indices, target, loops)
        elif isinstance(stmt, Evaluate):
            self.exprs(stmt, [stmt.value], path, loops, regions, write_path)

    def exprs(
        self,
        stmt: Stmt,
        exprs: Sequence[Expr],
        path: Path,
        loops: tuple[_Loop, ...],
        regions: dict[Buffer, tuple[Range, ...]],
        write_path: Path | None = None,
    ) -> None:
        for expr in exprs:
            for e in walk_expr(expr):
                if isinstance(e, BufferLoad):
                    self.point(e.buffer, e.indices, path, loops)
                elif isinstance(e, Call):
                    for i, arg in enumerate(e.args):
                        if isinstance(arg, AddressOf):
                            target = write_path if write_path is not None else path
                            self.pointer(e, i, arg, target, loops, regions)

    def point(
        self, buf: Buffer, indices: Sequence[Expr], path: Path, loops: tuple[_Loop, ...]
    ) -> None:
        starts = tuple(self.subst(i) for i in indices)
        self.out.uses.setdefault(buf, []).append(_Use(path, loops, starts, (1,) * len(indices)))

    def pointer(
        self,
        call: Call,
        position: int,
        arg: AddressOf,
        path: Path,
        loops: tuple[_Loop, ...],
        regions: dict[Buffer, tuple[Range, ...]],
    ) -> None:
        """A pointer passed to an intrinsic: it touches the region its block declares
        for the buffer or, failing that, the extents the intrinsic declares."""
        buf = arg.buffer
        region = regions.get(buf)
        starts: tuple[Expr, ...]
        extents: tuple[int | None, ...]
        if region is not None and all(isinstance(r.extent, IntImm) for r in region):
            starts = tuple(self.subst(r.min) for r in region)
            extents = tuple(r.extent.value for r in region)  # type: ignore[attr-defined]
        else:
            starts = tuple(self.subst(i) for i in arg.indices)
            trailing = self.extents_of(call, position)
            if trailing is None:
                extents = (None,) * len(starts)
            else:
                lead = len(starts) - len(trailing)
                extents = (*(1,) * lead, *trailing)
        self.out.uses.setdefault(buf, []).append(_Use(path, loops, starts, extents))

    def subst(self, expr: Expr) -> Expr:
        return substitute(expr, self.bindings) if self.bindings else expr


def _vars(expr: Expr) -> set[Var]:
    return {e for e in walk_expr(expr) if isinstance(e, Var)}


@dataclass(frozen=True)
class Compaction:
    """How one buffer was shrunk: index ``d`` becomes ``index[d] - offset[d]``.
    Offsets name grid axes by the variables in ``axes`` of ``plan_compaction``."""

    old: Buffer
    new: Buffer
    offsets: tuple[Affine, ...]


def plan_compaction(
    func: PrimFunc, extents_of: Callable[[Call, int], tuple[int, ...] | None]
) -> tuple[dict[Buffer, Compaction], _Axes]:
    """The compacted shape of every local and shared buffer ``func`` allocates, and
    the variable that stands for each grid axis in the offsets."""
    collector = _Collector(extents_of)
    collector.walk(func.body, (), (), None, {})
    out: dict[Buffer, Compaction] = {}
    for buf in collector.out.allocated:
        uses = collector.out.uses.get(buf)
        if buf.scope not in ("local", "shared") or not uses:
            continue
        compaction = _compact(buf, uses, collector.axes)
        if compaction.new.shape != buf.shape:
            out[buf] = compaction
    return out, collector.axes


def _compact(buf: Buffer, uses: list[_Use], axes: _Axes) -> Compaction:
    lca = _common_prefix([u.path for u in uses])
    shape: list[int] = []
    offsets: list[Affine] = []
    for dim, size in enumerate(buf.shape):
        regions = [_dim_region(buf, u, dim, lca, axes) for u in uses]
        known = [r for r in regions if r is not None]
        if len(known) != len(regions) or any(r[0] != known[0][0] for r in known):
            shape.append(size)
            offsets.append({None: 0})
            continue
        fixed = known[0][0]
        lo = min(r[1] for r in known)
        hi = max(r[2] for r in known)
        if hi - lo + 1 >= size:
            shape.append(size)
            offsets.append({None: 0})
            continue
        offset = dict(fixed)
        offset[None] = offset.get(None, 0) + lo
        shape.append(hi - lo + 1)
        offsets.append(_clean(offset))
    new = Buffer(buf.name, tuple(shape), buf.dtype, buf.scope)
    return Compaction(buf, new, tuple(offsets))


def _dim_region(
    buf: Buffer, use: _Use, dim: int, lca: Path, axes: _Axes
) -> tuple[Affine, int, int] | None:
    """(fixed part, lowest, highest) of the elements a use touches in ``dim``."""
    extent = use.extents[dim]
    if extent is None:
        return None
    ranges: dict[Var, tuple[int, int]] = {}
    fixed: set[Var] = set()
    canonical: dict[Var, Affine] = {}
    for loop in use.loops:
        inside = loop.path[: len(lca)] == lca and len(loop.path) >= len(lca)
        if buf.scope == "local":
            relaxed = inside and loop.tag is None
        else:
            relaxed = (inside and not _is_group(loop.tag)) or _is_thread(loop.tag)
        if relaxed:
            if loop.lo is None or loop.hi is None:
                return None
            ranges[loop.var] = (loop.lo, loop.hi)
        elif loop.tag is not None:
            canonical[loop.var] = _axis_form(loop, axes)
            fixed.add(axes[loop.tag])
        else:
            fixed.add(loop.var)
    expand = _expander(canonical)
    rest: Affine = {}
    lo = hi = 0
    for sign, term in terms(use.starts[dim]):
        form = affine(term, expand)
        if form is not None and all(v is None or v in ranges or v in fixed for v in form):
            for var, coef in form.items():
                coef *= sign
                if var is None:
                    lo, hi = lo + coef, hi + coef
                elif var in ranges:
                    a, b = coef * ranges[var][0], coef * ranges[var][1]
                    lo, hi = lo + min(a, b), hi + max(a, b)
                else:
                    rest[var] = rest.get(var, 0) + coef
            continue
        bounds = interval(term, ranges) if _vars(term) <= set(ranges) else None
        if bounds is None:
            return None
        a, b = sign * bounds[0], sign * bounds[1]
        lo, hi = lo + min(a, b), hi + max(a, b)
    return _clean(rest), lo, hi + extent - 1


def _expander(canonical: dict[Var, Affine]) -> Callable[[Var], Affine | None]:
    def expand(var: Var) -> Affine | None:
        return dict(canonical.get(var, {var: 1}))

    return expand


def terms(expr: Expr) -> list[tuple[int, Expr]]:
    """``expr`` as a signed sum: ``a + b - c`` is [(1, a), (1, b), (-1, c)]."""
    if isinstance(expr, Binary) and expr.op in ("add", "sub"):
        right = terms(expr.b)
        if expr.op == "sub":
            right = [(-sign, t) for sign, t in right]
        return terms(expr.a) + right
    return [(1, expr)]


def _is_thread(tag: str | None) -> bool:
    return tag is not None and tag.startswith("thread.")


def _is_group(tag: str | None) -> bool:
    return tag is not None and tag.startswith("threadgroup.")


def _common_prefix(paths: list[Path]) -> Path:
    first = paths[0]
    n = len(first)
    for p in paths[1:]:
        n = min(n, len(p))
        for i in range(n):
            if p[i] != first[i]:
                n = i
                break
    return first[:n]


def interval(expr: Expr, ranges: dict[Var, tuple[int, int]]) -> tuple[int, int] | None:
    """Integer bounds of ``expr`` given the ranges of its variables, or None."""
    if isinstance(expr, IntImm):
        return expr.value, expr.value
    if isinstance(expr, Var):
        return ranges.get(expr)
    if isinstance(expr, Cast) and expr.type.is_int and expr.value.dtype.is_int:
        return interval(expr.value, ranges)
    if not isinstance(expr, Binary):
        return None
    a = interval(expr.a, ranges)
    b = interval(expr.b, ranges)
    if a is None or b is None:
        return None
    if expr.op == "add":
        return a[0] + b[0], a[1] + b[1]
    if expr.op == "sub":
        return a[0] - b[1], a[1] - b[0]
    if expr.op == "mul":
        products = [x * y for x in a for y in b]
        return min(products), max(products)
    if expr.op in ("min", "max"):
        pick = min if expr.op == "min" else max
        return pick(a[0], b[0]), pick(a[1], b[1])
    if expr.op in ("floordiv", "floormod") and b[0] == b[1] and b[0] > 0:
        d = b[0]
        if expr.op == "floordiv":
            return a[0] // d, a[1] // d
        if a[0] // d == a[1] // d:
            return a[0] % d, a[1] % d
        return 0, d - 1
    return None


def _subtract(index: Expr, offset: Affine, canonical: dict[Var, Affine]) -> Expr:
    """``index - offset``, cancelling the terms of ``index`` that the offset covers."""
    fixed = {v for v in offset if v is not None}
    expand = _expander(canonical)
    kept: list[tuple[int, Expr]] = []
    covered: Affine = {None: 0}
    for sign, term in terms(index):
        form = affine(term, expand)
        if form is not None and all(v is None or v in fixed for v in form):
            for var, coef in form.items():
                covered[var] = covered.get(var, 0) + sign * coef
        else:
            kept.append((sign, term))
    if any(covered.get(v, 0) != c for v, c in offset.items() if v is not None):
        # The region analysis only compacts a dimension whose accesses all have
        # the offset's fixed part, so this means the two analyses disagree.
        raise RuntimeError(f"compaction offset {offset} does not match index {index}")
    out: Expr | None = None
    for sign, term in kept:
        if out is None:
            out = term if sign > 0 else Binary("sub", IntImm(0, index.dtype), term)
        else:
            out = Binary("add" if sign > 0 else "sub", out, term)
    const = covered.get(None, 0) - offset.get(None, 0)
    if out is None:
        return IntImm(const, index.dtype)
    return out + const if const else out


def compact_buffers(
    func: PrimFunc, extents_of: Callable[[Call, int], tuple[int, ...] | None]
) -> PrimFunc:
    """``func`` with every local and shared buffer shrunk to the region one thread
    or one threadgroup touches (see the module docstring)."""
    plan, axes = plan_compaction(func, extents_of)
    if not plan:
        return func
    return PrimFunc(func.name, func.params, _Rewriter(plan, axes).stmt(func.body), func.attrs)


class _Rewriter:
    def __init__(self, plan: dict[Buffer, Compaction], axes: _Axes) -> None:
        self.plan = plan
        self.axes = axes
        self.bindings: dict[Var, Expr] = {}  # block and let variables -> loop expressions
        self.canonical: dict[Var, Affine] = {}  # thread-bound loop variable -> axis form

    def index(self, buf: Buffer, indices: Sequence[Expr]) -> tuple[Buffer, tuple[Expr, ...]]:
        c = self.plan.get(buf)
        if c is None:
            return buf, tuple(self.expr(i) for i in indices)
        out: list[Expr] = []
        for index, offset in zip(indices, c.offsets, strict=True):
            index = self.expr(index)
            if offset == {None: 0}:
                out.append(index)
                continue
            # The offset is in loop variables, so the index is too: `by * 16 + ty`
            # minus `by * 16` reads as `ty`.
            index = substitute(index, self.bindings)
            out.append(_subtract(index, offset, self.canonical))
        return c.new, tuple(out)

    def expr(self, e: Expr) -> Expr:
        match e:
            case BufferLoad(buf, indices):
                return BufferLoad(*self.index(buf, indices))
            case AddressOf(buf, indices):
                return AddressOf(*self.index(buf, indices))
            case Binary(op, a, b):
                return Binary(op, self.expr(a), self.expr(b))
            case Compare(op, a, b):
                return Compare(op, self.expr(a), self.expr(b))
            case Logical(op, a, b):
                return Logical(op, self.expr(a), self.expr(b))
            case Not(a):
                return Not(self.expr(a))
            case Select(c, t, f):
                return Select(self.expr(c), self.expr(t), self.expr(f))
            case Cast(t, v):
                return Cast(t, self.expr(v))
            case Call(name, args, t):
                return Call(name, tuple(self.expr(a) for a in args), t)
            case _:
                return e

    def region(self, r: BufferRegion) -> BufferRegion:
        c = self.plan.get(r.buffer)
        if c is None:
            return r
        mins = self.index(r.buffer, [x.min for x in r.region])[1]
        return BufferRegion(
            c.new, tuple(Range(m, x.extent) for m, x in zip(mins, r.region, strict=True))
        )

    def stmt(self, s: Stmt) -> Stmt:
        match s:
            case SeqStmt(seq):
                return SeqStmt(tuple(self.stmt(x) for x in seq))
            case For():
                if s.kind is ForKind.THREAD_BINDING and s.thread_tag in self.axes:
                    lo = s.min.value if isinstance(s.min, IntImm) else 0
                    self.canonical[s.loop_var] = {self.axes[s.thread_tag]: 1, None: lo}
                return For(
                    s.loop_var,
                    self.expr(s.min),
                    self.expr(s.extent),
                    s.kind,
                    self.stmt(s.body),
                    s.thread_tag,
                    s.annotations,
                )
            case BlockRealize(values, predicate, block):
                for iv, value in zip(block.iter_vars, values, strict=True):
                    self.bindings[iv.var] = substitute(value, self.bindings)
                new_block = Block(
                    block.name,
                    block.iter_vars,
                    tuple(self.region(r) for r in block.reads),
                    tuple(self.region(r) for r in block.writes),
                    self.stmt(block.body),
                    None if block.init is None else self.stmt(block.init),
                    tuple(self.plan[b].new if b in self.plan else b for b in block.alloc_buffers),
                    block.annotations,
                )
                return BlockRealize(
                    tuple(self.expr(v) for v in values), self.expr(predicate), new_block
                )
            case BufferStore(buf, value, indices):
                new_buf, new_indices = self.index(buf, indices)
                return BufferStore(new_buf, self.expr(value), new_indices)
            case IfThenElse(cond, then, other):
                return IfThenElse(
                    self.expr(cond), self.stmt(then), None if other is None else self.stmt(other)
                )
            case LetStmt(var, value, body):
                self.bindings[var] = substitute(value, self.bindings)
                return LetStmt(var, self.expr(value), self.stmt(body))
            case Evaluate(value):
                return Evaluate(self.expr(value))
            case _:
                return s


# -- expression utilities ---------------------------------------------------------------


def walk_expr(expr: Expr) -> list[Expr]:
    """``expr`` and every expression below it, parents first."""
    out: list[Expr] = []
    stack = [expr]
    while stack:
        e = stack.pop()
        out.append(e)
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
    return out


def substitute(expr: Expr, mapping: dict[Var, Expr]) -> Expr:
    """``expr`` with variables replaced by their values in ``mapping``."""
    match expr:
        case Var():
            return mapping.get(expr, expr)
        case Binary(op, a, b):
            return Binary(op, substitute(a, mapping), substitute(b, mapping))
        case Compare(op, a, b):
            return Compare(op, substitute(a, mapping), substitute(b, mapping))
        case Logical(op, a, b):
            return Logical(op, substitute(a, mapping), substitute(b, mapping))
        case Not(a):
            return Not(substitute(a, mapping))
        case Select(c, t, f):
            return Select(substitute(c, mapping), substitute(t, mapping), substitute(f, mapping))
        case Cast(t, v):
            return Cast(t, substitute(v, mapping))
        case Call(name, args, t):
            return Call(name, tuple(substitute(a, mapping) for a in args), t)
        case BufferLoad(buf, indices):
            return BufferLoad(buf, tuple(substitute(i, mapping) for i in indices))
        case AddressOf(buf, indices):
            return AddressOf(buf, tuple(substitute(i, mapping) for i in indices))
        case _:
            return expr
