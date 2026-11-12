"""Tensorize: replace a loop nest by a registered intrinsic after matching its description."""

from __future__ import annotations

from tessera.tir import intrin
from tessera.tir.analysis import is_leaf_block
from tessera.tir.arith import Analyzer, Interval
from tessera.tir.expr import (
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
)
from tessera.tir.functor import free_vars, substitute, substitute_stmt
from tessera.tir.primitives.common import (
    ScheduleError,
    analyzer_for,
    conjoin,
    find_loop,
    loop_of,
    loop_roles,
    rebuild,
    with_body,
)
from tessera.tir.printer import script
from tessera.tir.stmt import (
    TRUE,
    Block,
    BlockRealize,
    Buffer,
    BufferRegion,
    BufferStore,
    For,
    PrimFunc,
    Range,
    SeqStmt,
    Stmt,
)

TENSORIZE = "tensorize"  # block annotation: the intrinsic's name
EXTENTS = "tensorize.extents"  # block annotation: the matched extents, comma separated


def _nest(stmt: Stmt) -> tuple[list[For], BlockRealize]:
    loops: list[For] = []
    while isinstance(stmt, For):
        loops.append(stmt)
        stmt = stmt.body
    if not isinstance(stmt, BlockRealize) or not is_leaf_block(stmt.block):
        raise ScheduleError("tensorize needs a perfect loop nest around a single block")
    return loops, stmt


class _Matcher:
    """Structural match of the description's body against the nest's body. Loads and
    stores match when the buffers correspond and the indices differ by offsets that
    do not depend on the matched loops."""

    def __init__(self, an: Analyzer, inner: set[Var]) -> None:
        self.an = an
        self.inner = inner
        self.buffers: dict[Buffer, Buffer] = {}
        self.offsets: dict[Buffer, list[Expr]] = {}
        self.accesses: dict[Buffer, list[tuple[Expr, ...]]] = {}
        self.stored: list[Buffer] = []

    def fail(self, why: str) -> ScheduleError:
        return ScheduleError(f"the loop nest does not match the intrinsic: {why}")

    def stmt(self, d: Stmt, a: Stmt) -> None:
        if isinstance(d, SeqStmt) and isinstance(a, SeqStmt) and len(d.seq) == len(a.seq):
            for x, y in zip(d.seq, a.seq, strict=True):
                self.stmt(x, y)
            return
        if isinstance(d, BufferStore) and isinstance(a, BufferStore):
            self.access(d.buffer, d.indices, a.buffer, a.indices)
            if d.buffer not in self.stored:
                self.stored.append(d.buffer)
            self.expr(d.value, a.value)
            return
        raise self.fail(f"{type(a).__name__} where the description has {type(d).__name__}")

    def access(
        self, dbuf: Buffer, didx: tuple[Expr, ...], abuf: Buffer, aidx: tuple[Expr, ...]
    ) -> None:
        mapped = self.buffers.setdefault(dbuf, abuf)
        if mapped is not abuf:
            raise self.fail(f"{dbuf.name} corresponds to both {mapped.name} and {abuf.name}")
        if abuf.dtype != dbuf.dtype or len(aidx) != len(didx):
            raise self.fail(f"{abuf.name} and {dbuf.name} differ in type or rank")
        offsets = [self.an.simplify(x - y) for x, y in zip(aidx, didx, strict=True)]
        for off in offsets:
            if free_vars(off) & self.inner:
                raise self.fail(f"the indices of {abuf.name} do not follow the description")
        known = self.offsets.setdefault(dbuf, offsets)
        if any(k != o for k, o in zip(known, offsets, strict=True)):
            raise self.fail(f"{abuf.name} is accessed at different offsets")
        self.accesses.setdefault(dbuf, []).append(didx)

    def expr(self, d: Expr, a: Expr) -> None:
        if type(d) is not type(a):
            raise self.fail(f"{type(a).__name__} where the description has {type(d).__name__}")
        if isinstance(d, IntImm | FloatImm):
            if d != a:
                raise self.fail("different constants")
        elif isinstance(d, Var):
            if d is not a:
                raise self.fail("a variable does not correspond to the description")
        elif isinstance(d, Binary | Compare | Logical):
            assert isinstance(a, Binary | Compare | Logical)
            if d.op != a.op:
                raise self.fail(f"{a.op} where the description has {d.op}")
            self.expr(d.a, a.a)
            self.expr(d.b, a.b)
        elif isinstance(d, Not):
            assert isinstance(a, Not)
            self.expr(d.a, a.a)
        elif isinstance(d, Select):
            assert isinstance(a, Select)
            self.expr(d.cond, a.cond)
            self.expr(d.true_value, a.true_value)
            self.expr(d.false_value, a.false_value)
        elif isinstance(d, Cast):
            assert isinstance(a, Cast)
            if d.type != a.type:
                raise self.fail("different casts")
            self.expr(d.value, a.value)
        elif isinstance(d, Call):
            assert isinstance(a, Call)
            if d.name != a.name or len(d.args) != len(a.args):
                raise self.fail(f"call {a.name} where the description has {d.name}")
            for x, y in zip(d.args, a.args, strict=True):
                self.expr(x, y)
        elif isinstance(d, BufferLoad):
            assert isinstance(a, BufferLoad)
            self.access(d.buffer, d.indices, a.buffer, a.indices)
        else:
            raise self.fail(f"cannot match {type(d).__name__}")


def _desc_nest(desc: PrimFunc) -> tuple[list[For], BlockRealize]:
    body = desc.body
    if isinstance(body, BlockRealize) and not body.block.iter_vars:
        body = body.block.body
    return _nest(body)


def tensorize(func: PrimFunc, loop: Var, name: str) -> PrimFunc:
    try:
        intrinsic = intrin.get(name)
    except KeyError as e:
        raise ScheduleError(str(e)) from None
    desc, impl = intrinsic.desc, intrinsic.impl
    if desc is None or impl is None:
        raise ScheduleError(f"intrinsic {name} has no description to tensorize with")
    lpath = find_loop(func, loop)
    loops, realize = _nest(loop_of(lpath))
    dloops, drealize = _desc_nest(desc)
    if len(loops) != len(dloops):
        raise ScheduleError(
            f"intrinsic {name} has {len(dloops)} loops, the nest at {loop.name} has {len(loops)}"
        )
    sizes: dict[str, int] = {}
    extents: list[int] = []
    an = analyzer_for(lpath)
    for a, d in zip(loops, dloops, strict=True):
        if not (isinstance(a.extent, IntImm) and a.min == IntImm(0) and d.min == IntImm(0)):
            raise ScheduleError(f"loop {a.loop_var.name} must start at 0 with a constant extent")
        n = a.extent.value
        if isinstance(d.extent, IntImm):
            if d.extent.value != n:
                raise ScheduleError(
                    f"loop {a.loop_var.name} runs {n} times, "
                    f"intrinsic {name} needs {d.extent.value}"
                )
        elif isinstance(d.extent, Var):
            if sizes.setdefault(d.extent.name, n) != n:
                raise ScheduleError(f"extent {d.extent.name} of intrinsic {name} matched twice")
        else:
            raise ScheduleError(f"intrinsic {name} has a loop extent that is not a constant")
        extents.append(n)
        an.bind(a.loop_var, Interval(0, n - 1))
    inner = {f.loop_var for f in loops}
    valid, outer_predicate = _edge_bounds(an, realize, loops, extents)
    if drealize.predicate != TRUE:
        raise ScheduleError(f"the description of {name} has a predicate")
    block, dblock = realize.block, drealize.block
    if (block.init is None) != (dblock.init is None):
        raise ScheduleError(
            "the block's reduction init does not match the description "
            "(decompose the reduction first)"
        )
    loop_map: dict[Var, Expr] = {d.loop_var: a.loop_var for d, a in zip(dloops, loops, strict=True)}
    abind = dict(zip((iv.var for iv in block.iter_vars), realize.iter_values, strict=True))
    dbind = {
        iv.var: substitute(v, loop_map)
        for iv, v in zip(dblock.iter_vars, drealize.iter_values, strict=True)
    }
    m = _Matcher(an, inner)
    if block.init is not None and dblock.init is not None:
        m.stmt(substitute_stmt(dblock.init, dbind), substitute_stmt(block.init, abind))
    m.stmt(substitute_stmt(dblock.body, dbind), substitute_stmt(block.body, abind))
    regions: list[BufferRegion] = []
    relaxed = {f.loop_var: Interval(0, n - 1) for f, n in zip(loops, extents, strict=True)}
    for p in desc.params:
        if p not in m.buffers:
            raise ScheduleError(f"parameter {p.name} of intrinsic {name} is never accessed")
        abuf = m.buffers[p]
        ranges: list[Range] = []
        for dim, off in enumerate(m.offsets[p]):
            clipped = _clipped_extent(p, dim, m.accesses[p], loops, valid)
            if clipped is not None:
                ranges.append(Range(an.simplify(off), clipped))
                continue
            los, his = [], []
            for idx in m.accesses[p]:
                lo, hi = an.relax(idx[dim], relaxed)
                if not (isinstance(lo, IntImm) and isinstance(hi, IntImm)):
                    raise ScheduleError(f"cannot bound the region of {p.name} in {name}")
                los.append(lo.value)
                his.append(hi.value)
            ranges.append(Range(an.simplify(off + min(los)), IntImm(max(his) - min(los) + 1)))
        regions.append(BufferRegion(abuf, tuple(ranges)))
    match = intrin.IntrinMatch(
        buffers=tuple(r.buffer for r in regions),
        pointers=tuple(AddressOf(r.buffer, tuple(x.min for x in r.region)) for r in regions),
        strides=tuple(r.buffer.strides for r in regions),
        regions=tuple(regions),
        extents=tuple(extents),
        valid=tuple(valid),
        sizes=sizes,
    )
    body = impl(match)
    writes = tuple(r for p, r in zip(desc.params, regions, strict=True) if p in m.stored)
    new_block = Block(
        block.name,
        (),
        tuple(regions),
        writes,
        body,
        None,
        (),
        (*block.annotations, (TENSORIZE, name), (EXTENTS, ",".join(map(str, extents)))),
    )
    return with_body(func, rebuild(lpath, BlockRealize((), outer_predicate, new_block)))


def _edge_bounds(
    an: Analyzer, realize: BlockRealize, loops: list[For], extents: list[int]
) -> tuple[list[Expr], Expr]:
    """The number of valid iterations of each matched loop, and the part of the
    block's predicate that does not involve them.

    A conjunct of the predicate over the matched loops must bound one spatial
    loop from above, ``A + k * x < c`` with ``A`` free of matched loops: the
    edge tile of an imperfect split. Then ``x`` runs ``(c - A + k - 1) // k``
    times, clipped to ``[0, extent]``. Any other condition (on a reduction loop,
    a lower bound, a relation between matched loops) is refused.
    """
    inner = {f.loop_var for f in loops}
    roles = dict(zip((f.loop_var for f in loops), loop_roles(realize, loops), strict=True))
    bounds: dict[Var, Expr] = {}
    outer: Expr = TRUE
    stack = [realize.predicate]
    while stack:
        c = stack.pop()
        if isinstance(c, Logical) and c.op == "and":
            stack.extend((c.a, c.b))
            continue
        used = free_vars(c) & inner
        if not used:
            outer = conjoin(outer, c)
            continue
        c = an.simplify(c)
        if c == TRUE:
            continue
        if any(roles[v] in ("R", "X") for v in used):
            raise ScheduleError("the block's predicate limits a reduction loop of the nest")
        if not (isinstance(c, Compare) and c.op == "lt" and isinstance(c.b, IntImm)):
            raise ScheduleError(f"the block's predicate {script(c)} is not an edge-tile bound")
        terms, const = an.linear(c.a)
        loop_terms = [(t, k) for t, k in terms.items() if free_vars(t) & inner]
        if len(loop_terms) != 1 or not isinstance(loop_terms[0][0], Var) or loop_terms[0][1] < 1:
            raise ScheduleError(f"the block's predicate {script(c)} is not an edge-tile bound")
        x, k = loop_terms[0]
        assert isinstance(x, Var)
        if roles[x] != "S":
            raise ScheduleError(f"the block's predicate {script(c)} bounds a loop it does not use")
        rest: Expr = IntImm(const)
        for t, coef in terms.items():
            if t is not x:
                rest = rest + t * coef
        count = an.simplify(Binary("floordiv", c.b - rest + (k - 1), IntImm(k)))
        bounds[x] = count if x not in bounds else an.simplify(Binary("min", bounds[x], count))
    valid: list[Expr] = []
    for f, n in zip(loops, extents, strict=True):
        if f.loop_var in bounds:
            clipped = Binary("min", Binary("max", bounds[f.loop_var], IntImm(0)), IntImm(n))
            valid.append(an.simplify(clipped))
        else:
            valid.append(IntImm(n))
    return valid, an.simplify(outer)


def _clipped_extent(
    param: Buffer,
    dim: int,
    accesses: list[tuple[Expr, ...]],
    loops: list[For],
    valid: list[Expr],
) -> Expr | None:
    """The extent of an edge tile along ``dim`` of ``param``: the valid count of the
    loop that indexes it, if that loop is cut by the predicate. A cut loop must
    index the dimension by itself (``c[i, j]``), so that the valid elements are a
    prefix of the tile."""
    cut = {
        f.loop_var: v
        for f, v in zip(loops, valid, strict=True)
        if not (isinstance(v, IntImm) and v == f.extent)
    }
    found: Expr | None = None
    for idx in accesses:
        e = idx[dim]
        hit = free_vars(e) & set(cut)
        if not hit:
            continue
        if not (isinstance(e, Var) and e in cut):
            raise ScheduleError(
                f"a loop cut by the predicate indexes {param.name} other than directly"
            )
        found = cut[e]
    return found
