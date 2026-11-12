"""Where intermediate buffers live in the generated C, and how big they are.

A block's ``alloc_buffers`` say which block owns a buffer, not where its
storage should be. Schedules create buffers with the shape of the whole
tensor (``cache_read``, ``cache_write``) and then move the blocks that use
them under tile loops; the copy made at each tile only needs the tile.

``plan_allocations`` decides, for every ``local`` buffer, the innermost
statement that encloses all of its uses (so a buffer used only inside a
parallel loop body gets one copy per iteration, on that thread's stack) and
shrinks it to the region those uses touch while the statement runs. A buffer
written by a reduction block is placed above the outermost loop of the
reduction, because its value is carried from one iteration of that loop to
the next. ``global`` intermediates keep their full shape and are allocated
once per kernel call, since threads may share them.

Statements are addressed by their path from the function body: the list of
child positions taken at each step (see ``children``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

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
    IfThenElse,
    IterKind,
    LetStmt,
    SeqStmt,
    Stmt,
)

type Path = tuple[int, ...]
# An affine form: coefficient per variable, the constant under ``None``.
type Affine = dict[Var | None, int]


def children(stmt: Stmt) -> list[Stmt]:
    """Child statements in path order. A realize's block body is child 0 and its
    init child 1."""
    if isinstance(stmt, For | LetStmt):
        return [stmt.body]
    if isinstance(stmt, SeqStmt):
        return list(stmt.seq)
    if isinstance(stmt, IfThenElse):
        return [stmt.then_case] if stmt.else_case is None else [stmt.then_case, stmt.else_case]
    if isinstance(stmt, BlockRealize):
        block = stmt.block
        return [block.body] if block.init is None else [block.body, block.init]
    if isinstance(stmt, Block):
        raise TypeError(f"block {stmt.name!r} outside a BlockRealize")
    return []


def own_exprs(stmt: Stmt) -> list[Expr]:
    """The expressions a statement evaluates itself (not those of its children)."""
    if isinstance(stmt, For):
        return [stmt.min, stmt.extent]
    if isinstance(stmt, BufferStore):
        return [*stmt.indices, stmt.value]
    if isinstance(stmt, Evaluate):
        return [stmt.value]
    if isinstance(stmt, IfThenElse):
        return [stmt.condition]
    if isinstance(stmt, LetStmt):
        return [stmt.value]
    if isinstance(stmt, BlockRealize):
        return [*stmt.iter_values, stmt.predicate]
    return []


def sub_exprs(expr: Expr) -> Iterator[Expr]:
    """``expr`` and every expression below it, parents first."""
    stack = [expr]
    while stack:
        e = stack.pop()
        yield e
        if isinstance(e, Binary | Compare | Logical):
            stack.extend((e.b, e.a))
        elif isinstance(e, Not):
            stack.append(e.a)
        elif isinstance(e, Select):
            stack.extend((e.false_value, e.true_value, e.cond))
        elif isinstance(e, Cast):
            stack.append(e.value)
        elif isinstance(e, Call):
            stack.extend(reversed(e.args))
        elif isinstance(e, BufferLoad | AddressOf):
            stack.extend(reversed(e.indices))


@dataclass(frozen=True)
class Compacted:
    """A local buffer shrunk to ``shape``; dimension ``d`` of an access becomes
    ``index[d] - offset[d]``, where ``offset[d]`` is an affine form over
    variables defined outside the allocation point."""

    buffer: Buffer
    path: Path  # the allocation point
    shape: tuple[int, ...]
    offset: tuple[Affine, ...]
    exact: tuple[bool, ...]  # False: that dimension kept its full extent

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


@dataclass
class AllocationPlan:
    local: dict[Path, list[Compacted]]  # allocated just before the statement at the path
    hoisted: list[Buffer]  # global intermediates, allocated at kernel entry
    bindings: dict[Var, Expr]  # block iteration variables and let variables -> value
    loop_ranges: dict[Var, tuple[int, int]]  # loops with constant bounds -> (min, max)
    defined_at: dict[Var, Path]  # where each variable is bound


@dataclass
class _Use:
    """One access: the statement path, and per dimension the first index and
    the number of elements (None: unknown, the whole dimension)."""

    path: Path
    starts: tuple[Expr, ...]
    extents: tuple[int | None, ...]


def plan_allocations(body: Stmt) -> AllocationPlan:
    """Place and size every buffer allocated by a block of ``body``."""
    walker = _UseWalker()
    walker.walk(body, (), [], None, {})
    plan = AllocationPlan({}, [], walker.bindings, walker.loop_ranges, walker.defined_at)
    for buf, decl_path in walker.declared:
        if buf.scope != "local":
            plan.hoisted.append(buf)
            continue
        uses = walker.uses.get(buf, [])
        if not uses:
            continue
        lca = _common_prefix([u.path for u in uses])
        if len(lca) < len(decl_path):
            lca = decl_path
        plan.local.setdefault(lca, []).append(_compact(buf, uses, lca, plan))
    return plan


type _Regions = dict[Buffer, tuple[tuple[Expr, int | None], ...] | None]


class _UseWalker:
    def __init__(self) -> None:
        self.uses: dict[Buffer, list[_Use]] = {}
        self.declared: list[tuple[Buffer, Path]] = []
        self.bindings: dict[Var, Expr] = {}
        self.loop_ranges: dict[Var, tuple[int, int]] = {}
        self.defined_at: dict[Var, Path] = {}
        self.loop_deps: dict[Var, frozenset[Var]] = {}  # block/let var -> loop vars it uses

    def walk(
        self,
        stmt: Stmt,
        path: Path,
        loops: list[tuple[Var, Path]],
        write_lift: Path | None,
        regions: _Regions,
    ) -> None:
        """Record the uses under ``stmt``. Writes inside a reduction block are
        recorded at ``write_lift``, the outermost loop of the reduction."""
        if isinstance(stmt, For):
            self.defined_at[stmt.loop_var] = path
            if isinstance(stmt.min, IntImm) and isinstance(stmt.extent, IntImm):
                lo = stmt.min.value
                self.loop_ranges[stmt.loop_var] = (lo, lo + stmt.extent.value - 1)
            self._record(stmt, path, write_lift, regions)
            inner = [*loops, (stmt.loop_var, path)]
            self.walk(stmt.body, (*path, 0), inner, write_lift, regions)
            return
        if isinstance(stmt, BlockRealize):
            block = stmt.block
            self._record(stmt, path, write_lift, regions)
            for buf in block.alloc_buffers:
                self.declared.append((buf, (*path, 0)))
            reduce_deps: set[Var] = set()
            for iv, value in zip(block.iter_vars, stmt.iter_values, strict=True):
                self.bindings[iv.var] = value
                self.defined_at[iv.var] = path
                self.loop_deps[iv.var] = self._deps(value)
                if iv.kind is IterKind.REDUCE:
                    reduce_deps |= self.loop_deps[iv.var]
            for var, loop_path in loops:
                if var in reduce_deps:
                    if write_lift is None or len(loop_path) < len(write_lift):
                        write_lift = loop_path
                    break
            block_regions = _block_regions(block)
            for i, child in enumerate(children(stmt)):
                self.walk(child, (*path, i), loops, write_lift, block_regions)
            return
        if isinstance(stmt, LetStmt):
            self.bindings[stmt.var] = stmt.value
            self.defined_at[stmt.var] = path
            self.loop_deps[stmt.var] = self._deps(stmt.value)
        self._record(stmt, path, write_lift, regions)
        for i, child in enumerate(children(stmt)):
            self.walk(child, (*path, i), loops, write_lift, regions)

    def _record(self, stmt: Stmt, path: Path, write_lift: Path | None, regions: _Regions) -> None:
        write_path = write_lift if write_lift is not None else path
        if isinstance(stmt, BufferStore):
            self._point(stmt.buffer, write_path, stmt.indices)
        for expr in own_exprs(stmt):
            for e in sub_exprs(expr):
                if isinstance(e, BufferLoad):
                    self._point(e.buffer, path, e.indices)
                elif isinstance(e, AddressOf):
                    # A microkernel reads and may write the region its block
                    # declares for the buffer, starting at the pointer.
                    region = regions.get(e.buffer)
                    if region is None:
                        starts = tuple(e.indices)
                        extents: tuple[int | None, ...] = tuple(None for _ in e.indices)
                    else:
                        starts = tuple(r[0] for r in region)
                        extents = tuple(r[1] for r in region)
                    self.uses.setdefault(e.buffer, []).append(_Use(write_path, starts, extents))

    def _point(self, buf: Buffer, path: Path, indices: tuple[Expr, ...]) -> None:
        self.uses.setdefault(buf, []).append(_Use(path, indices, tuple(1 for _ in indices)))

    def _deps(self, expr: Expr) -> frozenset[Var]:
        deps: set[Var] = set()
        for e in sub_exprs(expr):
            if isinstance(e, Var):
                deps |= self.loop_deps.get(e, frozenset({e}))
        return frozenset(deps)


def _block_regions(block: Block) -> _Regions:
    """The region of each buffer the block declares, if it is a single box with
    constant extents."""
    out: _Regions = {}
    for region in (*block.writes, *block.reads):
        dims = tuple(
            (r.min, r.extent.value if isinstance(r.extent, IntImm) else None) for r in region.region
        )
        if region.buffer in out and out[region.buffer] != dims:
            out[region.buffer] = None
        else:
            out[region.buffer] = dims
    return out


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


def _is_inner(var: Var, lca: Path, plan: AllocationPlan) -> bool:
    where = plan.defined_at.get(var)
    return where is not None and where[: len(lca)] == lca


def _compact(buf: Buffer, uses: list[_Use], lca: Path, plan: AllocationPlan) -> Compacted:
    shape: list[int] = []
    offsets: list[Affine] = []
    exact: list[bool] = []
    for dim, size in enumerate(buf.shape):
        regions = [_dim_region(u.starts[dim], u.extents[dim], lca, plan) for u in uses]
        known = [r for r in regions if r is not None]
        if len(known) != len(regions) or any(r[0] != known[0][0] for r in known):
            shape.append(size)
            offsets.append({})
            exact.append(False)
            continue
        rest = known[0][0]
        lo = min(r[1] for r in known)
        hi = max(r[2] for r in known)
        extent = min(hi - lo + 1, size)
        offset = dict(rest)
        offset[None] = offset.get(None, 0) + lo
        shape.append(max(extent, 1))
        offsets.append({k: v for k, v in offset.items() if v != 0})
        exact.append(True)
    return Compacted(buf, lca, tuple(shape), tuple(offsets), tuple(exact))


def _dim_region(
    index: Expr, extent: int | None, lca: Path, plan: AllocationPlan
) -> tuple[Affine, int, int] | None:
    """(outer part, lowest, highest) of the elements ``index .. index + extent - 1``
    over the inner variables."""
    if extent is None:
        return None
    region = _start_region(index, lca, plan)
    if region is None:
        return None
    return region[0], region[1], region[2] + extent - 1


def _start_region(index: Expr, lca: Path, plan: AllocationPlan) -> tuple[Affine, int, int] | None:
    form = affine(index, lambda v: _expand_inner(v, lca, plan))
    if form is not None:
        rest: Affine = {}
        lo = hi = form.get(None, 0)
        for var, coef in form.items():
            if var is None:
                continue
            if _is_inner(var, lca, plan):
                bounds = plan.loop_ranges.get(var)
                if bounds is None:
                    return None
                a, b = coef * bounds[0], coef * bounds[1]
                lo += min(a, b)
                hi += max(a, b)
            else:
                rest[var] = coef
        return rest, lo, hi
    interval = _interval(index, lca, plan)
    if interval is None:
        return None
    return {}, interval[0], interval[1]


def _expand_inner(var: Var, lca: Path, plan: AllocationPlan) -> Affine | None:
    if _is_inner(var, lca, plan) and var in plan.bindings:
        return affine(plan.bindings[var], lambda v: _expand_inner(v, lca, plan))
    return {var: 1}


def affine(expr: Expr, expand: Callable[[Var], Affine | None]) -> Affine | None:
    """``expr`` as a linear combination of variables plus a constant, or None.
    ``expand`` gives the form of each variable (itself, or its binding)."""
    if isinstance(expr, IntImm):
        return {None: expr.value}
    if isinstance(expr, Var):
        return expand(expr)
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
        return {k: v for k, v in out.items() if v != 0 or k is None}
    return None


def _scale(form: Affine, factor: int) -> Affine:
    return {k: v * factor for k, v in form.items() if v * factor != 0 or k is None}


def _interval(expr: Expr, lca: Path, plan: AllocationPlan) -> tuple[int, int] | None:
    """Integer bounds of ``expr`` when every variable in it is defined inside
    the allocation point."""
    if isinstance(expr, IntImm):
        return expr.value, expr.value
    if isinstance(expr, Var):
        if not _is_inner(expr, lca, plan):
            return None
        if expr in plan.loop_ranges:
            return plan.loop_ranges[expr]
        if expr in plan.bindings:
            return _interval(plan.bindings[expr], lca, plan)
        return None
    if isinstance(expr, Cast) and expr.type.is_int and expr.value.dtype.is_int:
        return _interval(expr.value, lca, plan)
    if not isinstance(expr, Binary):
        return None
    a = _interval(expr.a, lca, plan)
    b = _interval(expr.b, lca, plan)
    if a is None or b is None:
        return None
    if expr.op == "add":
        return a[0] + b[0], a[1] + b[1]
    if expr.op == "sub":
        return a[0] - b[1], a[1] - b[0]
    if expr.op == "mul":
        products = [x * y for x in a for y in b]
        return min(products), max(products)
    if expr.op == "min":
        return min(a[0], b[0]), min(a[1], b[1])
    if expr.op == "max":
        return max(a[0], b[0]), max(a[1], b[1])
    if expr.op in ("floordiv", "floormod") and b[0] == b[1] and b[0] > 0:
        d = b[0]
        if expr.op == "floordiv":
            return a[0] // d, a[1] // d
        if a[0] // d == a[1] // d:
            return a[0] % d, a[1] % d
        return 0, d - 1
    return None


def compacted_index(c: Compacted, dim: int, index: Expr, plan: AllocationPlan) -> Expr:
    """Index ``dim`` of an access to the compacted buffer ``c``."""
    if not c.exact[dim]:
        return index
    form = affine(index, lambda v: _expand_inner(v, c.path, plan))
    if form is None:
        # Bounded by interval arithmetic, so the offset is a constant.
        shift = c.offset[dim].get(None, 0)
        return index if shift == 0 else Binary("sub", index, IntImm(shift, index.dtype))
    out = dict(form)
    for k, v in c.offset[dim].items():
        out[k] = out.get(k, 0) - v
    return affine_to_expr(out)


def affine_to_expr(form: Affine) -> Expr:
    """``sum(coef * var) + constant``, in the form's order."""
    expr: Expr | None = None
    for var, coef in form.items():
        if var is None or coef == 0:
            continue
        term: Expr = var if coef == 1 else Binary("mul", var, IntImm(coef, var.dtype))
        expr = term if expr is None else Binary("add", expr, term)
    const = form.get(None, 0)
    if expr is None:
        return IntImm(const)
    if const > 0:
        return Binary("add", expr, IntImm(const, expr.dtype))
    if const < 0:
        return Binary("sub", expr, IntImm(-const, expr.dtype))
    return expr
