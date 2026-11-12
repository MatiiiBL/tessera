"""GPU schedule templates: one per pattern of ``tessera.plan.PATTERNS``.

A template takes a ``Schedule`` of one lowered group and asks its ``Config``
for every decision, with defaults chosen from measurements on an M4 (see
docs/metal.md). Every template maps the computation onto threadgroups and
threads with ``bind``; the code generator turns the result into MSL.

- **injective**: every block inlined into the output, its loops fused and
  split over threadgroups, threads and a few elements per thread.
- **matmul**, **conv2d**: a threadgroup computes a tile of the output, each
  thread a register tile of it (``registers``), with the accumulators in
  local memory and the epilogue applied as they are stored. Tiles of A and B
  can be staged in threadgroup memory (``shared``) every ``tile_k`` steps of
  the reduction. A convolution is the same computation for each output row:
  the row's pixels as rows, output channels as columns, ``(kh, kw, c)`` as the
  reduction. A matrix-vector product (few rows) runs one thread per output
  element instead (``strategy``).
- **reduce**, **softmax**, **layernorm**: either one threadgroup per row with
  its threads splitting each reduction (combined with SIMD-group functions
  and threadgroup memory), or one thread per row when the rows are the
  contiguous dimension (``strategy``).
- **pool**, **depthwise_conv2d**: one thread per output element, the window
  loops unrolled.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial

from tessera.plan import LoweredGroup
from tessera.tir.expr import Var
from tessera.tir.functor import walk_stmts
from tessera.tir.schedule import Schedule, ScheduleError
from tessera.tir.stmt import Block, BlockRealize, BufferStore, IterKind, PrimFunc
from tessera.tune.space import Config

type MetalTemplate = Callable[[Schedule, LoweredGroup, Config], None]

THREAD_COUNTS = (32, 64, 128, 256, 512, 1024)


# -- knobs --------------------------------------------------------------------------------


def knob_int(cfg: Config, name: str, options: Sequence[int], default: int) -> int:
    value = cfg.choice(name, list(options), default=default)
    assert isinstance(value, int)
    return value


def knob_pair(
    cfg: Config, name: str, options: Sequence[tuple[int, int]], default: tuple[int, int]
) -> tuple[int, int]:
    value = cfg.choice(name, list(options), default=default)
    assert isinstance(value, tuple)
    assert len(value) == 2
    return value[0], value[1]


def knob_bool(cfg: Config, name: str, default: bool) -> bool:
    value = cfg.choice(name, [default, not default], default=default)
    assert isinstance(value, bool)
    return value


def knob_str(cfg: Config, name: str, options: Sequence[str], default: str) -> str:
    value = cfg.choice(name, list(options), default=default)
    assert isinstance(value, str)
    return value


# -- looking at the schedule ------------------------------------------------------------


def find_block(func: PrimFunc, name: str) -> BlockRealize:
    """The realize of the block called ``name``."""
    for s in walk_stmts(func.body):
        if isinstance(s, BlockRealize) and s.block.name == name:
            return s
    raise KeyError(name)


def is_reduction(block: Block) -> bool:
    return any(iv.kind is IterKind.REDUCE for iv in block.iter_vars)


def output_block(sch: Schedule) -> str:
    """The block that writes the function's output (its last parameter)."""
    out = sch.func.params[-1]
    for name in reversed(sch.blocks()):
        block = find_block(sch.func, name).block
        if any(isinstance(s, BufferStore) and s.buffer is out for s in walk_stmts(block.body)):
            return name
    return sch.blocks()[-1]


def split_loops(sch: Schedule, block: str) -> tuple[list[Var], list[Var]]:
    """The loops around ``block`` that drive spatial and reduction variables."""
    realize = find_block(sch.func, block)
    reduce_vars: set[Var] = set()
    from tessera.tir.functor import free_vars

    for iv, value in zip(realize.block.iter_vars, realize.iter_values, strict=True):
        if iv.kind is IterKind.REDUCE:
            reduce_vars |= free_vars(value)
    loops = sch.get_loops(block)
    return [v for v in loops if v not in reduce_vars], [v for v in loops if v in reduce_vars]


def attempt(sch: Schedule, step: Callable[[Schedule], object]) -> bool:
    """Apply ``step`` if it succeeds on a copy of the schedule; whether it did."""
    try:
        step(sch.copy())
    except ScheduleError:
        return False
    step(sch)
    return True


def succeeds(sch: Schedule, step: Callable[[Schedule], object]) -> bool:
    """Whether ``step`` applies to a copy of the schedule (the schedule is unchanged)."""
    try:
        step(sch.copy())
    except ScheduleError:
        return False
    return True


def fuse_all(sch: Schedule, loops: Sequence[Var]) -> Var | None:
    if not loops:
        return None
    return sch.fuse(*loops) if len(loops) > 1 else loops[0]


def inline_all(sch: Schedule, keep: set[str]) -> None:
    """Inline every block outside ``keep`` that can be inlined into its consumers."""
    for name in sch.blocks():
        if name in keep or is_reduction(find_block(sch.func, name).block):
            continue
        if not attempt(sch, partial(Schedule.compute_inline, block=name)):
            attempt(sch, partial(Schedule.reverse_compute_inline, block=name))


def threads_for(extent: int, cap: int = 1024) -> list[int]:
    """Thread counts worth trying for a loop of ``extent``: up to the first count
    that covers it."""
    out = [t for t in THREAD_COUNTS if t <= cap]
    covering = [t for t in out if t >= extent]
    return [t for t in out if t <= (covering[0] if covering else out[-1])]


# -- injective ---------------------------------------------------------------------------


def schedule_injective(sch: Schedule, lowered: LoweredGroup, cfg: Config) -> None:
    """Elementwise, broadcast and injective groups: one fused loop over the output."""
    out = output_block(sch)
    inline_all(sch, {out})
    fused = fuse_all(sch, sch.get_loops(out))
    if fused is None:
        return
    total = sch.extent(fused)
    options = threads_for(total)
    threads = knob_int(cfg, "threads", options, min(256, options[-1]))
    per_thread = knob_int(cfg, "per_thread", [1, 2, 4], 1)
    # Element e of a thread is threads apart from its element e + 1, so the threads
    # of a SIMD group touch consecutive addresses at every step.
    group, element, thread = sch.split(fused, [None, per_thread, threads])
    sch.bind(group, "threadgroup.x")
    sch.bind(thread, "thread.x")
    if per_thread > 1:
        sch.unroll(element)


# -- matmul and convolutions ------------------------------------------------------------

THREAD_SHAPES = ((8, 8), (4, 16), (16, 4), (16, 8), (8, 16), (16, 16), (2, 32), (1, 64))
REGISTER_TILES = ((1, 1), (1, 4), (1, 8), (2, 2), (4, 2), (2, 4), (4, 4), (8, 4), (4, 8))


def _tile_threads(
    sch: Schedule, loops: Sequence[Var], threads: tuple[int, int], regs: tuple[int, int]
) -> None:
    """``loops`` (rows, columns) cover a ``threads x regs`` tile: bind the thread
    parts and unroll the register parts."""
    rows, cols = loops
    rt, rr = sch.split(rows, [threads[0], regs[0]])
    ct, cr = sch.split(cols, [threads[1], regs[1]])
    sch.reorder(rt, ct, rr, cr)
    sch.bind(rt, "thread.y")
    sch.bind(ct, "thread.x")
    sch.unroll(rr)
    sch.unroll(cr)


def _stage(sch: Schedule, block: str, index: int, at: Var, threads: tuple[int, int]) -> None:
    """Copy the main block's ``index``-th input into threadgroup memory at ``at``,
    the threads of the group sharing the copy."""
    cache = sch.cache_read(block, index, "shared")
    sch.compute_at(cache, at)
    loops = sch.get_loops(cache)
    rows, cols = loops[-2:]
    # Splitting rows and columns separately (not fused) keeps every index affine,
    # which is what lets the schedule prove the threads write different elements.
    ro, ry = sch.split(rows, [None, threads[0]])
    co, cx = sch.split(cols, [None, threads[1]])
    sch.reorder(ro, co, ry, cx)
    sch.bind(ry, "thread.y")
    sch.bind(cx, "thread.x")


@dataclass(frozen=True)
class GemmKnobs:
    strategy: str  # "tiles", "elements" or "simdgroup"
    threads: tuple[int, int]
    registers: tuple[int, int]
    tile_k: int
    shared: bool
    element_threads: int
    simd_groups: tuple[int, int] = (1, 1)  # SIMD groups per threadgroup (rows, columns)
    simd_tiles: tuple[int, int] = (1, 1)  # 8 x 8 accumulator tiles per SIMD group
    image_rows: int = 1  # convolution: output rows per threadgroup, on thread.z
    simd_k: int = 32  # convolution: input channels per call of the SIMD-group microkernel


SIMD_GROUPS = ((1, 1), (1, 2), (2, 1), (2, 2), (1, 4), (4, 1), (2, 4), (4, 2))
# Tile layouts in order of preference (from measurements on transformer and square
# shapes); the first that divides the problem is the default.
SIMD_PREFERENCE = (
    ((1, 2), (2, 4)),
    ((2, 1), (2, 4)),
    ((1, 1), (2, 4)),
    ((1, 2), (2, 2)),
    ((1, 1), (2, 2)),
    ((1, 1), (1, 4)),
    ((1, 1), (1, 2)),
    ((1, 1), (1, 1)),
)


def simd_layout(m: int, n: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """The preferred SIMD-group layout whose threadgroup tile divides the ``m`` rows
    (and, if one does, the ``n`` columns). Columns may end in a partial tile."""
    rows = [(g, t) for g, t in SIMD_PREFERENCE if m % (8 * g[0] * t[0]) == 0]
    whole = [(g, t) for g, t in rows if n % (8 * g[1] * t[1]) == 0]
    choices = whole or rows
    return choices[0] if choices else None


def gemm_knobs(
    cfg: Config,
    m: int,
    n: int,
    k: int,
    outputs: int,
    can_stage: bool,
    simd: tuple[tuple[int, int], tuple[int, int]] | None = None,
    *,
    image_rows: Sequence[int] = (1,),
    simd_k: Sequence[int] = (32,),
) -> GemmKnobs:
    """Every knob of the GEMM templates, asked whatever the strategy. ``simd`` is
    the SIMD-group layout to default to, when that strategy applies; a
    convolution also chooses among ``image_rows`` output rows per threadgroup.
    Small rows (a 7 x 7 image) get several rows per threadgroup and a pixel tile
    no wider than the row."""
    default = "simdgroup" if simd else "elements" if m <= 4 else "tiles"
    strategy = knob_str(cfg, "strategy", ["tiles", "elements", "simdgroup"], default)
    shapes = [s for s in THREAD_SHAPES if s[0] <= max(m, 1) * 2 and s[1] <= max(n, 32)]
    default_threads = (8, 8) if (8, 8) in shapes else shapes[0]
    threads = knob_pair(cfg, "threads", shapes, default_threads)
    row_tile = 4 if m > 16 else 2 if m > 8 else 1
    regs = knob_pair(cfg, "registers", REGISTER_TILES, (row_tile, 4))
    stack_default = 1 if m > 16 or len(image_rows) == 1 else 2 if m > 8 else 4
    stack = knob_int(cfg, "image_rows", image_rows, stack_default)
    tile_k = knob_int(
        cfg, "tile_k", [t for t in (4, 8, 16, 32) if t <= max(k, 4)], min(8, max(k, 4))
    )
    stage = knob_bool(cfg, "shared", can_stage) and can_stage
    options = threads_for(outputs)
    element_threads = knob_int(cfg, "element_threads", options, min(128, options[-1]))
    groups, tiles = simd or SIMD_PREFERENCE[0]
    from tessera.backend.metal.microkernels import FRAGMENT_TILES

    simd_groups = knob_pair(cfg, "simd_groups", SIMD_GROUPS, groups)
    simd_tiles = knob_pair(cfg, "simd_tiles", FRAGMENT_TILES, tiles)
    channel_tile = knob_int(cfg, "simd_k", simd_k, 32 if 32 in simd_k else simd_k[-1])
    return GemmKnobs(
        strategy,
        threads,
        regs,
        tile_k,
        stage,
        element_threads,
        simd_groups,
        simd_tiles,
        stack,
        channel_tile,
    )


def _extent(sch: Schedule, loops: Sequence[Var]) -> int:
    out = 1
    for v in loops:
        out *= sch.extent(v)
    return out


def _gemm(
    sch: Schedule,
    lowered: LoweredGroup,
    knobs: GemmKnobs,
    *,
    batch: Sequence[Var],
    rows: Sequence[Var],
    cols: Sequence[Var],
    reduction: Sequence[Var],
    staged: Sequence[int] = (),
    stack: Var | None = None,
    window: Sequence[Var] = (),
) -> None:
    """Tile an output of ``rows x cols`` (each a list of loops to fuse) reduced
    over ``reduction``, as described in the module docstring; ``staged`` are the
    main block's reads that may be copied to threadgroup memory, ``stack``
    (a convolution's output rows) is split so that ``knobs.image_rows`` of its
    iterations share a threadgroup, on thread.z, and ``window`` (a convolution's
    kernel loops) are reduction loops kept whole outside the tiled one."""
    main = lowered.main_block
    assert main is not None
    if knobs.strategy == "elements":
        _per_element(sch, lowered, knobs.element_threads, unroll=False)
        return
    if knobs.strategy == "simdgroup":
        _simdgroup(sch, lowered, knobs, batch=batch, rows=rows, cols=cols, reduction=reduction)
        return
    m_loop, n_loop, k_loop = fuse_all(sch, rows), fuse_all(sch, cols), fuse_all(sch, reduction)
    assert m_loop is not None
    assert n_loop is not None
    assert k_loop is not None
    threads, regs = knobs.threads, knobs.registers
    (ty, tx), (rm, rn) = threads, regs

    mo, mt, mr = sch.split(m_loop, [None, ty, rm])
    no, nt, nr = sch.split(n_loop, [None, tx, rn])
    ko, ki = sch.split(k_loop, [None, knobs.tile_k])
    stacked: list[Var] = []
    if stack is not None and knobs.image_rows > 1:
        so, si = sch.split(stack, [None, knobs.image_rows])
        batch, stacked = [*batch, so], [si]
    elif stack is not None:
        batch = [*batch, stack]
    # The reduction steps outside the thread loops, so that the tile copies and
    # the accumulators' init and store can each get their own thread loops.
    sch.reorder(*batch, mo, no, *window, ko, *stacked, mt, nt, ki, mr, nr)
    for loop in stacked:
        sch.bind(loop, "thread.z")
    batch_loop = fuse_all(sch, batch)
    if batch_loop is not None:
        sch.bind(batch_loop, "threadgroup.z")
    sch.bind(mo, "threadgroup.y")
    sch.bind(no, "threadgroup.x")

    store = sch.cache_write(main, 0, "local")
    sch.reverse_compute_at(store, no)
    if lowered.epilogue_block is not None:
        sch.reverse_compute_inline(lowered.epilogue_block)
    store_loops = sch.get_loops(store)
    below = store_loops[store_loops.index(no) + 1 :]
    if len(below) < 2 + len(stacked):
        # A tile with a single row has no row loop in its store to give the threads
        # that hold each row (matrix-vector products use the "elements" strategy).
        raise ScheduleError("the output tile has a single row")
    if stacked:
        # The store's loops below the tile: rows of the image, pixels, channels.
        sch.bind(below[-3], "thread.z")
    _tile_threads(sch, below[-2:], threads, regs)

    if knobs.shared:
        for index in staged:
            attempt(sch, partial(_stage, block=main, index=index, at=ko, threads=threads))
    init = sch.decompose_reduction(main, window[0] if window else ko)
    # The init's loops mirror the update's below ko: threads, then registers.
    *_, it, jt, ir, jr = sch.get_loops(init)
    if stacked:
        sch.bind(sch.get_loops(init)[-5], "thread.z")
    sch.bind(it, "thread.y")
    sch.bind(jt, "thread.x")
    sch.unroll(ir)
    sch.unroll(jr)
    sch.bind(mt, "thread.y")
    sch.bind(nt, "thread.x")
    sch.unroll(mr)
    sch.unroll(nr)


def _simdgroup(
    sch: Schedule,
    lowered: LoweredGroup,
    knobs: GemmKnobs,
    *,
    batch: Sequence[Var],
    rows: Sequence[Var],
    cols: Sequence[Var],
    reduction: Sequence[Var],
) -> None:
    """SIMD-group matrices: each SIMD group computes an ``8 fm x 8 fn`` tile with
    the ``metal.simdgroup_gemm`` microkernel over the whole reduction, loading A
    and B from device memory. The accumulators start and end in threadgroup
    memory, where the lanes of the group set the init and apply the epilogue."""
    from tessera.backend.metal import microkernels

    main = lowered.main_block
    assert main is not None
    (gm, gn), (fm, fn) = knobs.simd_groups, knobs.simd_tiles
    if len(rows) != 1 or len(cols) != 1 or len(reduction) != 1:
        raise ScheduleError("SIMD-group matrices need one row, column and reduction loop")
    (m_loop,), (n_loop,), (k_loop,) = rows, cols, reduction
    m, n, k = sch.extent(m_loop), sch.extent(n_loop), sch.extent(k_loop)
    # Columns may end in a partial tile: the kernel computes the whole tile (its
    # loads past the last column stay in the allocation's padding) and only the
    # valid columns are stored. A partial tile of rows would read whole rows of A
    # past the end, and the reduction is walked 8 at a time.
    if m % (8 * gm * fm) or k % 8 or n < 8 * fn:
        raise ScheduleError(f"{m} x {n} x {k} does not fit {8 * fm} x {8 * fn} x 8 tiles")
    mo, ms, mi = sch.split(m_loop, [None, gm, 8 * fm])
    no, ns, ni = sch.split(n_loop, [None, gn, 8 * fn])
    sch.reorder(mo, no, ms, ns, mi, ni, k_loop)
    batch_loop = fuse_all(sch, batch)
    if batch_loop is not None:
        sch.bind(batch_loop, "threadgroup.z")
    sch.bind(mo, "threadgroup.y")
    sch.bind(no, "threadgroup.x")
    sch.bind(ms, "thread.z")
    sch.bind(ns, "thread.y")
    store = sch.cache_write(main, 0, "shared")
    sch.reverse_compute_at(store, ns)
    if lowered.epilogue_block is not None:
        sch.reverse_compute_inline(lowered.epilogue_block)
    _lanes(sch, store)
    init = sch.decompose_reduction(main, mi)
    _lanes(sch, init)
    sch.tensorize(mi, microkernels.name(fm, fn, _gemm_ranks(sch, main)))


def _gemm_ranks(sch: Schedule, block: str) -> tuple[int, int, int]:
    """The ranks of A, B and C in a block ``C[...] = C[...] + A[...] * B[...]``."""
    from tessera.tir.expr import Binary, BufferLoad
    from tessera.tir.stmt import BufferStore

    body = find_block(sch.func, block).block.body
    if (
        isinstance(body, BufferStore)
        and isinstance(body.value, Binary)
        and body.value.op == "add"
        and isinstance(body.value.b, Binary)
        and body.value.b.op == "mul"
        and isinstance(body.value.b.a, BufferLoad)
        and isinstance(body.value.b.b, BufferLoad)
    ):
        a, b = body.value.b.a, body.value.b.b
        return a.buffer.ndim, b.buffer.ndim, body.buffer.ndim
    raise ScheduleError(f"block {block} is not C = C + A * B")


def _lanes(sch: Schedule, block: str) -> None:
    """Spread a SIMD group's tile (the block's last two loops) over its 32 lanes."""
    fused = sch.fuse(*sch.get_loops(block)[-2:])
    _, lane = sch.split(fused, [None, 32])
    sch.bind(lane, "thread.x")


def schedule_matmul(sch: Schedule, lowered: LoweredGroup, cfg: Config) -> None:
    """MatMul (batched or not) with its prologue and epilogue: SIMD-group matrices
    when the operands are plain matrices and the shape divides into their tiles,
    otherwise tiles of scalar code, or for a matrix-vector product (few rows) one
    thread per output element."""
    main = lowered.main_block
    assert main is not None
    spatial, reduction = split_loops(sch, main)
    *batch, rows, cols = spatial
    staged = _affine_reads(find_block(sch.func, main).block)
    m, n, k = sch.extent(rows), sch.extent(cols), _extent(sch, reduction)
    simd = simd_layout(m, n) if k % 8 == 0 else None
    if simd is not None:
        # The default must apply: try the SIMD-group schedule on a copy.
        trial = GemmKnobs("simdgroup", (8, 8), (4, 4), 8, False, 32, *simd)
        if not succeeds(sch, partial(_simdgroup_check, lowered=lowered, knobs=trial)):
            simd = None
    knobs = gemm_knobs(cfg, m, n, k, _extent(sch, spatial), bool(staged), simd)
    _gemm(
        sch,
        lowered,
        knobs,
        batch=batch,
        rows=[rows],
        cols=[cols],
        reduction=reduction,
        staged=staged,
    )


def _simdgroup_check(sch: Schedule, *, lowered: LoweredGroup, knobs: GemmKnobs) -> None:
    """Apply the SIMD-group schedule to ``sch`` from scratch (for ``attempt``)."""
    main = lowered.main_block
    assert main is not None
    spatial, reduction = split_loops(sch, main)
    *batch, rows, cols = spatial
    _simdgroup(sch, lowered, knobs, batch=batch, rows=[rows], cols=[cols], reduction=reduction)


def _affine_reads(block: Block) -> list[int]:
    """The reads of ``block`` that touch one element per iteration at an affine
    index: their tile at one step of the reduction is a box, which threadgroup
    memory can hold compactly. A prologue that reshapes (``k // 64``, ``k % 64``)
    makes its read a gather, and two reads of one buffer (Q and K sliced from one
    projection) merge into a wide region."""
    from tessera.backend.metal.lower import affine
    from tessera.tir.expr import IntImm

    return [
        i
        for i, region in enumerate(block.reads)
        if all(
            affine(r.min) is not None and isinstance(r.extent, IntImm) and r.extent.value == 1
            for r in region.region
        )
    ]


def schedule_conv2d(sch: Schedule, lowered: LoweredGroup, cfg: Config) -> None:
    """Conv2d over NHWC/HWIO as a GEMM per output row: each output row's pixels
    ``ow`` are the rows, output channels the columns and ``(kh, kw, c)`` the
    reduction, with the window loops ``kh, kw`` outside the tiled channel loop
    (so the input's coordinates and padding test do not change inside it).

    With SIMD-group matrices (stride 1, input channels a multiple of 8), each
    SIMD group computes its tile of the window's input (the padding block, or a
    copy of the input) into threadgroup memory, then runs the microkernel over
    ``simd_k`` input channels for every ``(kh, kw)``. Otherwise scalar tiles:
    tiling ``ow`` rather than the fused pixel index keeps every index affine, so
    a thread's accumulators stay a small register tile, and a threadgroup takes
    ``image_rows`` consecutive output rows (on thread.z) so that small images
    still give it enough work."""
    main = lowered.main_block
    assert main is not None
    spatial, reduction = split_loops(sch, main)
    *batch, rows, pixels, channels = spatial
    *window, in_channels = reduction
    m, n, k = sch.extent(pixels), sch.extent(channels), sch.extent(in_channels)
    staged = _affine_reads(find_block(sch.func, main).block)
    channel_tiles = [t for t in (8, 16, 32, 64) if k % t == 0] or [32]
    # One SIMD group per 8 x 32 or 16 x 32 tile; two across the output channels
    # when there are channels for both.
    layout = ((1, 2 if n > 32 else 1), (1 if m <= 8 else 2, 4))
    channel_tile = 32 if 32 in channel_tiles else channel_tiles[-1]
    trial = GemmKnobs("simdgroup", (8, 8), (4, 4), 8, False, 32, *layout, 1, channel_tile)
    applies = succeeds(sch, partial(_simdgroup_conv_check, lowered=lowered, knobs=trial))
    simd = layout if applies else None
    knobs = gemm_knobs(
        cfg,
        m,
        n,
        k,
        _extent(sch, spatial),
        bool(staged),
        simd,
        image_rows=(1, 2, 4, 8),
        simd_k=channel_tiles,
    )
    if knobs.strategy == "simdgroup":
        _simdgroup_conv(sch, lowered, knobs)
        return
    # The scalar schedules read the input through the padding test: inline a
    # padding block back into the convolution.
    inline_all(sch, {main, *([lowered.epilogue_block] if lowered.epilogue_block else [])})
    spatial, reduction = split_loops(sch, main)
    *batch, rows, pixels, channels = spatial
    *window, in_channels = reduction
    _gemm(
        sch,
        lowered,
        knobs,
        batch=batch,
        rows=[pixels],
        cols=[channels],
        reduction=[in_channels],
        staged=_affine_reads(find_block(sch.func, main).block),
        stack=rows,
        window=window,
    )


def _simdgroup_conv(sch: Schedule, lowered: LoweredGroup, knobs: GemmKnobs) -> None:
    """The SIMD-group schedule of ``schedule_conv2d`` (see its docstring)."""
    from tessera.backend.metal import microkernels

    main = lowered.main_block
    assert main is not None
    spatial, reduction = split_loops(sch, main)
    *batch, rows, pixels, channels = spatial
    *window, in_channels = reduction
    (gm, gn), (fm, fn), kt = knobs.simd_groups, knobs.simd_tiles, knobs.simd_k
    if sch.extent(in_channels) % kt or kt % 8:
        raise ScheduleError(f"{sch.extent(in_channels)} input channels are not tiles of {kt}")
    ranks = _gemm_ranks(sch, main)
    source = _conv_input(sch, main)
    mo, ms, mi = sch.split(pixels, [None, gm, 8 * fm])
    no, ns, ni = sch.split(channels, [None, gn, 8 * fn])
    ko, ki = sch.split(in_channels, [None, kt])
    sch.reorder(*batch, rows, mo, no, ms, ns, *window, ko, mi, ni, ki)
    group = fuse_all(sch, [*batch, rows])
    assert group is not None
    sch.bind(group, "threadgroup.z")
    sch.bind(mo, "threadgroup.y")
    sch.bind(no, "threadgroup.x")
    sch.bind(ms, "thread.z")
    sch.bind(ns, "thread.y")
    store = sch.cache_write(main, 0, "shared")
    sch.reverse_compute_at(store, ns)
    if lowered.epilogue_block is not None:
        sch.reverse_compute_inline(lowered.epilogue_block)
    _lanes(sch, store)
    # Each SIMD group's input tile for one (kh, kw, channel tile) step, in
    # threadgroup memory: rows past the image's edge never read device memory.
    if isinstance(source, str):
        sch.compute_inline(sch.cache_write(source, 0, "shared"))
        sch.compute_at(source, ko)
        _lanes(sch, source)
    else:
        copy = sch.cache_read(main, source, "shared")
        sch.compute_at(copy, ko)
        _lanes(sch, copy)
    init = sch.decompose_reduction(main, window[0] if window else ko)
    _lanes(sch, init)
    sch.tensorize(mi, microkernels.name(fm, fn, ranks, int(lowered.dims.get("SW", 1))))


def _conv_input(sch: Schedule, main: str) -> str | int:
    """The convolution's input: the block that pads it (a block of this function),
    or the index of the main block's read of it."""
    from tessera.tir.expr import Binary, BufferLoad

    block = find_block(sch.func, main).block
    body = block.body
    a = None
    if isinstance(body, BufferStore) and isinstance(body.value, Binary):
        product = body.value.b
        if isinstance(product, Binary) and isinstance(product.a, BufferLoad):
            a = product.a
    if a is None:
        # The input is read through a padding test: not a matrix the microkernel takes.
        raise ScheduleError(f"block {main} does not read its input as a plain load")
    for name in sch.blocks():
        realize = find_block(sch.func, name)
        if any(
            isinstance(s, BufferStore) and s.buffer is a.buffer
            for s in walk_stmts(realize.block.body)
        ):
            return name
    return next(i for i, r in enumerate(block.reads) if r.buffer is a.buffer)


def _simdgroup_conv_check(sch: Schedule, *, lowered: LoweredGroup, knobs: GemmKnobs) -> None:
    """Apply the SIMD-group convolution schedule to ``sch`` (for ``succeeds``)."""
    _simdgroup_conv(sch, lowered, knobs)


# -- reductions ----------------------------------------------------------------------------


def _row_loops(
    sch: Schedule, block: str, lowered: LoweredGroup
) -> tuple[list[Var], list[Var], list[Var]]:
    """The output block's spatial loops as (outer, reduced axis, inner), matched
    against the group's OUTER R INNER sizes from the innermost loop out. The
    reduced axis is empty when the output does not have it (a reduction)."""
    spatial, _ = split_loops(sch, block)
    extents = [sch.extent(v) for v in spatial]
    outer_size = int(lowered.dims.get("OUTER", 1))
    r_size = int(lowered.dims.get("R", 1))
    inner_size = int(lowered.dims.get("INNER", 1))
    total = 1
    for e in extents:
        total *= e
    has_axis = total == outer_size * r_size * inner_size and r_size > 1

    def take(size: int, end: int) -> int:
        product, i = 1, end
        while i > 0 and product < size:
            i -= 1
            product *= extents[i]
        if product != size:
            raise ScheduleError(f"no loops of {block} cover {size} elements")
        return i

    inner_start = take(inner_size, len(spatial)) if inner_size > 1 else len(spatial)
    r_start = take(r_size, inner_start) if has_axis else inner_start
    return spatial[:r_start], spatial[r_start:inner_start], spatial[inner_start:]


def schedule_rows(sch: Schedule, lowered: LoweredGroup, cfg: Config) -> None:
    """Reduce, Softmax and LayerNorm: reductions over R for each of OUTER x INNER rows."""
    out = output_block(sch)
    reductions = [
        b for b in sch.blocks() if b != out and is_reduction(find_block(sch.func, b).block)
    ]
    inline_all(sch, {out, *reductions})
    outer, axis, inner = _row_loops(sch, out, lowered)
    r_size = int(lowered.dims.get("R", 1))
    inner_size = int(lowered.dims.get("INNER", 1))
    outer_size = int(lowered.dims.get("OUTER", 1))
    # Every knob is asked whatever the strategy, so that the space is the same
    # for every point of it.
    strategy = knob_str(
        cfg, "strategy", ["rows", "columns"], "columns" if inner_size >= 32 else "rows"
    )
    options = threads_for(r_size)
    threads = knob_int(cfg, "threads", options, min(256, options[-1]))
    per_group = knob_int(cfg, "rows_per_group", [1, 2, 4, 8], 1)
    column_options = threads_for(outer_size * inner_size)
    column_threads = knob_int(cfg, "column_threads", column_options, min(128, column_options[-1]))
    if strategy == "columns":
        _rows_as_columns(sch, out, reductions, column_threads)
        return
    if threads * per_group > 1024:
        raise ScheduleError(f"{threads} threads x {per_group} rows exceed 1024 threads")
    if axis and inner:
        sch.reorder(*outer, *inner, *axis)
    row = fuse_all(sch, [*outer, *inner])
    anchor: Var | None = None
    if row is not None:
        group, anchor = sch.split(row, [None, per_group])
        sch.bind(group, "threadgroup.x")
        if per_group > 1:
            sch.bind(anchor, "thread.y")

    def lanes(loops: Sequence[Var]) -> None:
        fused = fuse_all(sch, loops)
        if fused is not None:
            _, lane = sch.split(fused, [None, threads])
            sch.bind(lane, "thread.x")

    lanes(split_loops(sch, out)[1] if is_reduction(find_block(sch.func, out).block) else axis)
    # Last statistic first: each one's consumers must already be under the row loop.
    for red in reversed(reductions):
        # The reduced value lives in threadgroup memory: one thread stores it and
        # every thread of the row reads it.
        copy = sch.cache_write(red, 0, "shared")
        sch.compute_inline(copy)
        if anchor is not None:
            sch.compute_at(red, anchor)
        lanes(split_loops(sch, red)[1])


def _registers(sch: Schedule, blocks: Sequence[str]) -> None:
    """Keep each block's result in a thread-private buffer instead of device memory."""
    for block in blocks:
        sch.compute_inline(sch.cache_write(block, 0, "local"))


def _rows_as_columns(sch: Schedule, out: str, reductions: Sequence[str], threads: int) -> None:
    """One thread per row, reducing serially: for rows that are the contiguous
    dimension, consecutive threads read consecutive addresses."""
    _registers(sch, reductions)
    spatial, _ = split_loops(sch, out)
    fused = fuse_all(sch, spatial)
    if fused is None:
        return
    group, thread = sch.split(fused, [None, threads])
    sch.bind(group, "threadgroup.x")
    sch.bind(thread, "thread.x")
    for red in reversed(reductions):
        sch.compute_at(red, thread)


# -- pooling and depthwise convolution ----------------------------------------------------


def schedule_window(sch: Schedule, lowered: LoweredGroup, cfg: Config) -> None:
    """Pooling and depthwise convolution: one thread per output element, the
    window loops unrolled."""
    out = output_block(sch)
    options = threads_for(_extent(sch, split_loops(sch, out)[0]))
    threads = knob_int(cfg, "threads", options, min(128, options[-1]))
    unroll = knob_bool(cfg, "unroll_window", True)
    _per_element(sch, lowered, threads, unroll=unroll)


def _per_element(sch: Schedule, lowered: LoweredGroup, threads: int, *, unroll: bool) -> None:
    """One thread per element of the output: the main block's reduction runs in the
    thread, its result kept in a register for the epilogue."""
    out = output_block(sch)
    main = lowered.main_block
    inline_all(sch, {out} | ({main} if main else set()))
    fused = fuse_all(sch, split_loops(sch, out)[0])
    if fused is None:
        return
    reducing = [main] if main and main != out else []
    _registers(sch, reducing)
    group, thread = sch.split(fused, [None, threads])
    sch.bind(group, "threadgroup.x")
    sch.bind(thread, "thread.x")
    for red in reducing:
        sch.compute_at(red, thread)
    if unroll:
        for block in [*reducing, out]:
            for loop in split_loops(sch, block)[1]:
                sch.unroll(loop)


TEMPLATES: Mapping[str, MetalTemplate] = {
    "conv2d": schedule_conv2d,
    "depthwise_conv2d": schedule_window,
    "matmul": schedule_matmul,
    "pool": schedule_window,
    "reduce": schedule_rows,
    "softmax": schedule_rows,
    "layernorm": schedule_rows,
    "injective": schedule_injective,
}
