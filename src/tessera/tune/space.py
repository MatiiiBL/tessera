"""Tunable knobs for schedule templates.

A template is written once against a ``Config``. It asks for each decision by
name::

    def schedule_matmul(sch, group, cfg):
        mo, mi = sch.split(m, cfg.split("tile_m", M, parts=2))
        if cfg.choice("unroll_k", [False, True]):
            ...

Run with a ``DefineConfig``, the template records every knob and its options
(the search space) and gets each knob's default. Run with an ``ApplyConfig``,
it gets the values of one point in the space. ``space_of`` builds the space by
running a template once on a copy of the schedule.
"""

from __future__ import annotations

import itertools
import json
import random
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import prod
from typing import Literal

type KnobValue = int | str | bool | tuple[int, ...]
type SplitPolicy = Literal["factors", "power2"]


class Config:
    """What a template asks for its decisions."""

    def split(
        self,
        name: str,
        extent: int,
        *,
        parts: int = 2,
        policy: SplitPolicy = "factors",
        max_inner: int | None = None,
        default: tuple[int, ...] | None = None,
    ) -> tuple[int, ...]:
        """Factors for splitting a loop of ``extent`` into ``parts`` loops,
        outermost first. ``factors``: exact factorizations. ``power2``: inner
        factors are powers of two and the outermost covers the rest (the split
        may be imperfect). ``max_inner`` caps the innermost factor."""
        options = split_options(extent, parts, policy, max_inner)
        return self._decide(name, options, default)  # type: ignore[return-value]

    def choice(
        self, name: str, options: Sequence[KnobValue], *, default: KnobValue | None = None
    ) -> KnobValue:
        return self._decide(name, list(options), default)

    def _decide(self, name: str, options: list[KnobValue], default: KnobValue | None) -> KnobValue:
        raise NotImplementedError


class DefineConfig(Config):
    """Records the knobs a template asks for; answers with defaults."""

    def __init__(self) -> None:
        self.knobs: dict[str, list[KnobValue]] = {}
        self.defaults: dict[str, KnobValue] = {}

    def _decide(self, name: str, options: list[KnobValue], default: KnobValue | None) -> KnobValue:
        if not options:
            raise ValueError(f"knob {name!r} has no options")
        if name in self.knobs:
            raise ValueError(f"knob {name!r} is defined twice")
        value = default if default is not None else options[0]
        if value not in options:
            raise ValueError(f"default {value!r} of knob {name!r} is not an option")
        self.knobs[name] = options
        self.defaults[name] = value
        return value


class ApplyConfig(Config):
    """Answers with the values of one point of the space."""

    def __init__(self, values: Mapping[str, KnobValue]) -> None:
        self.values = dict(values)

    def _decide(self, name: str, options: list[KnobValue], default: KnobValue | None) -> KnobValue:
        if name not in self.values:
            raise KeyError(f"config has no value for knob {name!r}")
        value = self.values[name]
        if value not in options:
            raise ValueError(f"{value!r} is not an option of knob {name!r}")
        return value


@dataclass(frozen=True)
class ConfigSpace:
    """The cartesian product of a template's knobs, in definition order."""

    knobs: Mapping[str, Sequence[KnobValue]]
    defaults: Mapping[str, KnobValue]

    @property
    def size(self) -> int:
        return prod(len(v) for v in self.knobs.values())

    def get(self, index: int) -> dict[str, KnobValue]:
        """The point with this index (mixed-radix, last knob fastest)."""
        if not 0 <= index < self.size:
            raise IndexError(index)
        values: dict[str, KnobValue] = {}
        for name, options in reversed(list(self.knobs.items())):
            index, i = divmod(index, len(options))
            values[name] = options[i]
        return {name: values[name] for name in self.knobs}

    def index_of(self, values: Mapping[str, KnobValue]) -> int:
        index = 0
        for name, options in self.knobs.items():
            index = index * len(options) + list(options).index(values[name])
        return index

    def default(self) -> dict[str, KnobValue]:
        return dict(self.defaults)

    def sample(self, rng: random.Random) -> dict[str, KnobValue]:
        return {name: rng.choice(list(options)) for name, options in self.knobs.items()}

    def __iter__(self) -> Iterator[dict[str, KnobValue]]:
        names = list(self.knobs)
        for combo in itertools.product(*self.knobs.values()):
            yield dict(zip(names, combo, strict=True))


def space_of(template: Callable[[Config], None]) -> ConfigSpace:
    """Run ``template`` (a closure over a schedule copy) once to record its knobs."""
    cfg = DefineConfig()
    template(cfg)
    return ConfigSpace(cfg.knobs, cfg.defaults)


def config_json(values: Mapping[str, KnobValue]) -> str:
    """Canonical JSON for a point (tuples become lists), for logs and cache keys."""
    return json.dumps(
        {k: list(v) if isinstance(v, tuple) else v for k, v in values.items()}, sort_keys=True
    )


def config_from_json(text: str) -> dict[str, KnobValue]:
    raw = json.loads(text)
    return {k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()}


def split_options(
    extent: int, parts: int, policy: SplitPolicy = "factors", max_inner: int | None = None
) -> list[KnobValue]:
    """Every way to split ``extent`` into ``parts`` factors (outermost first)."""
    if extent < 1 or parts < 1:
        raise ValueError(f"cannot split extent {extent} into {parts} parts")
    if policy == "factors":
        options = list(_factorizations(extent, parts))
    else:
        powers = [p for p in (2**i for i in range(extent.bit_length() + 1)) if p <= extent]
        options = []
        for inner in itertools.product(powers, repeat=parts - 1):
            covered = prod(inner)
            if covered <= extent:
                options.append((-(-extent // covered), *inner))
    if max_inner is not None:
        options = [f for f in options if f[-1] <= max_inner] or [options[0]]
    return sorted(set(options), key=lambda f: tuple(reversed(f)))  # type: ignore[arg-type]


def _factorizations(extent: int, parts: int) -> Iterator[tuple[int, ...]]:
    if parts == 1:
        yield (extent,)
        return
    for d in range(1, extent + 1):
        if extent % d == 0:
            for rest in _factorizations(extent // d, parts - 1):
                yield (d, *rest)
