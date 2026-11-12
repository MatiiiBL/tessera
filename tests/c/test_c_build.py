"""The clang build cache."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tessera.backend.c import build as cbuild
from tessera.target import Target

PROGRAM = '#include <stdio.h>\nint main(void) { printf("%d\\n", VALUE); return 0; }\n'


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("TESSERA_CACHE", str(tmp_path))
    return tmp_path / "c"


def _program(value: int) -> dict[str, str]:
    return {"main.c": PROGRAM.replace("VALUE", str(value))}


def test_builds_into_the_cache_and_reuses_it(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = cbuild.build(_program(7), "main.c", Target(), kind="executable")
    assert first.parent.parent == cache
    assert (first.parent / "main.c").read_text() == _program(7)["main.c"]
    assert subprocess.run([str(first)], capture_output=True, text=True).stdout == "7\n"

    def no_compiler(*args: object, **kwargs: object) -> None:
        raise AssertionError("a cached build ran the compiler")

    monkeypatch.setattr(cbuild.subprocess, "run", no_compiler)
    assert cbuild.build(_program(7), "main.c", Target(), kind="executable") == first


def test_sources_and_flags_change_the_key(cache: Path) -> None:
    a = cbuild.build(_program(1), "main.c", Target(), kind="executable")
    b = cbuild.build(_program(2), "main.c", Target(), kind="executable")
    c = cbuild.build(_program(1), "main.c", Target(cflags=("-DUNUSED=1",)), kind="executable")
    assert len({a.parent, b.parent, c.parent}) == 3
    assert sorted(p.name for p in cache.iterdir()) == sorted(
        p.parent.name for p in (a, b, c)
    )  # no temporary directories left behind


def test_compile_errors_carry_the_diagnostics(cache: Path) -> None:
    with pytest.raises(cbuild.BuildError, match="undeclared"):
        cbuild.build({"main.c": "int main(void) { return x; }\n"}, "main.c", Target())
    assert not any(cache.iterdir())


def test_missing_compiler_is_a_build_error(cache: Path) -> None:
    with pytest.raises(cbuild.BuildError, match="cannot run"):
        cbuild.build(_program(1), "main.c", Target(cc="no-such-cc-tessera"))


def test_isa_flags() -> None:
    assert cbuild.isa_flags(Target(isa="generic")) == []
    assert cbuild.isa_flags(Target(isa="sme")) == ["-mcpu=apple-m4"]
    flags = cbuild.compile_flags(Target(isa="generic", cflags=("-DX",)))
    assert flags[:2] == ["-std=c11", "-O3"]
    assert flags.index("-DX") < flags.index("-shared")
    assert "-ffinite-math-only" not in flags
    assert "-ffast-math" not in flags
