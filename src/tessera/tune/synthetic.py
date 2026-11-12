"""Synthetic tuning tasks with an analytic cost, for testing and comparing searches.

``SyntheticMatmul`` is a tiled matrix multiplication whose configuration
picks three-level tiles for M and N, a two-level split of K, the position of
the outer K loop, unrolling and vectorization. ``schedule`` builds the loop
nest a CPU template would produce for that configuration (so the real
feature extractor sees real loop nests), and ``measure`` returns the time a
simple machine model predicts instead of running anything. The model counts
cycles per step of the register tile (FMA and load throughput, register
spills, loop overhead, unrolling), adds penalties when the tiles overflow
the L1 and L2 caches, divides the work over the threads that the outer loop
can feed, and adds a deterministic ±1.5% perturbation so ties are broken
the way noisy measurements break them.

The landscape has the features that make real tuning hard: most of the space
is several times slower than the best point, the good region is narrow
(vectorized register tiles that fit in registers, outer extents that feed
every thread, panels that fit in L1), and the knobs interact. Because the
cost is a pure function of the configuration, a search with a fixed seed is
reproducible, and the optimum of a small space can be found exhaustively.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from math import ceil

from tessera.backend.base import Prebuilder, get_backend
from tessera.dtype import DType
from tessera.target import Target
from tessera.tir.expr import FloatImm, IntImm, Var
from tessera.tir.stmt import (
    TRUE,
    Block,
    BlockRealize,
    Buffer,
    BufferRegion,
    BufferStore,
    For,
    ForKind,
    IterKind,
    IterVar,
    PrimFunc,
    Range,
    Stmt,
)
from tessera.tune.space import Config, ConfigSpace, DefineConfig, KnobValue, config_json

CLOCK_HZ = 3.5e9
L1_BYTES = 128 * 1024
L2_BYTES = 4 * 1024 * 1024
VECTOR_REGISTERS = 32
LANES = 4


@dataclass(frozen=True)
class SyntheticMatmul:
    """``C[M, N] = A[M, K] @ B[K, N]`` with a machine model instead of a machine."""

    m: int = 64
    n: int = 64
    k: int = 64
    threads: int = 4

    @property
    def key(self) -> str:
        return f"synthetic-matmul:{self.m}x{self.n}x{self.k}"

    @property
    def target_key(self) -> str:
        return f"synthetic-t{self.threads}"

    @cached_property
    def space(self) -> ConfigSpace:
        cfg = DefineConfig()
        self.define(cfg)
        return ConfigSpace(cfg.knobs, cfg.defaults)

    def define(self, cfg: Config) -> dict[str, KnobValue]:
        """Ask ``cfg`` for every knob, as a template would."""
        return {
            "tile_m": cfg.split("tile_m", self.m, parts=3),
            "tile_n": cfg.split("tile_n", self.n, parts=3),
            "tile_k": cfg.split("tile_k", self.k, parts=2),
            "k_outer": cfg.choice("k_outer", [False, True]),
            "unroll_k": cfg.choice("unroll_k", [False, True]),
            "vectorize": cfg.choice("vectorize", [False, True]),
        }

    def schedule(self, config: Mapping[str, KnobValue]) -> PrimFunc:
        """The tiled loop nest of ``config``."""
        mo, mt, mi = _split(config["tile_m"])
        no, nt, ni = _split(config["tile_n"])
        ko, ki = _split(config["tile_k"])
        extents = {
            "i_outer": mo,
            "j_outer": no,
            "k_outer": ko,
            "i_tile": mt,
            "j_tile": nt,
            "k_inner": ki,
            "i_inner": mi,
            "j_inner": ni,
        }
        loops = {name: Var(name) for name in extents}
        kinds = dict.fromkeys(loops, ForKind.SERIAL)
        kinds["i_outer"] = ForKind.PARALLEL
        if config["unroll_k"]:
            kinds["k_inner"] = ForKind.UNROLLED
        if config["vectorize"]:
            kinds["j_inner"] = ForKind.VECTORIZED
        tiles = (
            ["k_outer", "i_tile", "j_tile"]
            if config["k_outer"]
            else ["i_tile", "j_tile", "k_outer"]
        )
        order = ["i_outer", "j_outer", *tiles, "k_inner", "i_inner", "j_inner"]

        a = Buffer("A", (self.m, self.k))
        b = Buffer("B", (self.k, self.n))
        c = Buffer("C", (self.m, self.n))
        i, j, kk = Var("i"), Var("j"), Var("k")
        v = loops
        bind_i = (v["i_outer"] * mt + v["i_tile"]) * mi + v["i_inner"]
        bind_j = (v["j_outer"] * nt + v["j_tile"]) * ni + v["j_inner"]
        bind_k = v["k_outer"] * ki + v["k_inner"]
        block = Block(
            name="C",
            iter_vars=(
                IterVar(i, Range.of(self.m)),
                IterVar(j, Range.of(self.n)),
                IterVar(kk, Range.of(self.k), IterKind.REDUCE),
            ),
            reads=(BufferRegion.full(c), BufferRegion.full(a), BufferRegion.full(b)),
            writes=(BufferRegion.full(c),),
            body=BufferStore(c, c[i, j] + a[i, kk] * b[kk, j], (i, j)),
            init=BufferStore(c, FloatImm(0.0, DType.float32), (i, j)),
        )
        body: Stmt = BlockRealize((bind_i, bind_j, bind_k), TRUE, block)
        for name in reversed(order):
            body = For(v[name], IntImm(0), IntImm(extents[name]), kinds[name], body)
        root = Block("root", (), (), (), body)
        return PrimFunc("matmul", (a, b, c), BlockRealize((), TRUE, root))

    def build(self, config: Mapping[str, KnobValue]) -> dict[str, KnobValue]:
        return dict(config)

    def measure(self, built: Mapping[str, KnobValue]) -> float:
        return self.cost(built)

    def cost(self, config: Mapping[str, KnobValue]) -> float:
        """Seconds the machine model predicts for ``config``."""
        mo, mt, mi = _split(config["tile_m"])
        no, nt, ni = _split(config["tile_n"])
        ko, ki = _split(config["tile_k"])
        lanes = LANES if config["vectorize"] else 1
        nv = ceil(ni / lanes)  # vector registers per row of the register tile
        fmas, loads = mi * nv, mi + nv
        cycles_per_step = max(fmas, loads) / 2.0 + 1.0  # two FMA pipes, two load ports
        registers = fmas + nv + 1
        if registers > VECTOR_REGISTERS:
            cycles_per_step *= 1.0 + 0.15 * (registers - VECTOR_REGISTERS)
        if config["unroll_k"]:
            cycles_per_step += -0.8 if ki * fmas <= 256 else 0.3 * cycles_per_step
        steps = (self.m // mi) * (self.n // ni) * self.k
        cycles = steps * cycles_per_step

        panel = 4 * (mi * ki + ki * ni + mi * ni)
        if panel > L1_BYTES // 2:
            cycles *= 1.6
        if config["k_outer"]:
            block = 4 * (mt * mi * ki + ki * nt * ni + mt * mi * nt * ni)
            cycles += 4 * self.m * self.n * ko * 2 / 16  # C tile reread once per K block
        else:
            block = 4 * (mi * self.k + self.k * nt * ni)
        if block > L2_BYTES // self.threads:
            cycles *= 1.8
        elif block > L1_BYTES:
            cycles *= 1.25

        rounds = ceil(mo / self.threads)
        cycles = cycles / mo * rounds  # the outer loop is split over the threads
        cycles += 2000.0 * rounds + 20.0 * mo * no * ko  # dispatch and loop overhead
        return cycles / CLOCK_HZ * (1.0 + 0.03 * (_unit_hash(config) - 0.5))

    def optimum(self) -> tuple[dict[str, KnobValue], float]:
        """The best configuration by exhaustive search (small spaces only)."""
        best = min(self.space, key=self.cost)
        return best, self.cost(best)


def _split(value: KnobValue) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"expected split factors, got {value!r}")
    return value


def _unit_hash(config: Mapping[str, KnobValue]) -> float:
    digest = hashlib.sha256(config_json(config).encode()).digest()
    return int.from_bytes(digest[:8], "little") / 2**64


@dataclass(frozen=True)
class MeasuredMatmul:
    """The loop nests of a ``SyntheticMatmul``, timed on a real backend.

    The same space and the same functions, but ``build`` and ``measure`` go
    through the target's backend (``prebuild`` and ``measure``), so the
    tuner's process runner and search are exercised against real timings
    without needing a lowered group or a template.
    """

    shape: SyntheticMatmul
    target: Target
    repeat: int = 3
    min_seconds: float = 0.02

    @property
    def key(self) -> str:
        return self.shape.key.replace("synthetic-matmul", "matmul-loops")

    @property
    def target_key(self) -> str:
        return self.target.key

    @property
    def space(self) -> ConfigSpace:
        return self.shape.space

    def schedule(self, config: Mapping[str, KnobValue]) -> PrimFunc:
        return self.shape.schedule(config)

    def build(self, config: Mapping[str, KnobValue]) -> PrimFunc:
        func = self.schedule(config)
        backend = get_backend(self.target.kind)
        if isinstance(backend, Prebuilder):
            backend.prebuild(func, self.target)
        return func

    def measure(self, built: PrimFunc) -> float:
        backend = get_backend(self.target.kind)
        return backend.measure(built, self.target, repeat=self.repeat, min_seconds=self.min_seconds)
