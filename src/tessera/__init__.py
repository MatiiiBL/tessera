"""Tessera: a tensor compiler for Apple silicon.

``tessera.compile(model, target=...)`` compiles an ONNX model; see
``tessera.driver.compile`` for the options.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.1.0"

if TYPE_CHECKING:
    from tessera.driver import compile as compile


def __getattr__(name: str) -> object:
    # Every tessera module imports this package first; loading the driver (and
    # with it the whole pipeline) only when ``compile`` is used avoids cycles.
    if name == "compile":
        from tessera.driver import compile

        return compile
    raise AttributeError(f"module 'tessera' has no attribute {name!r}")


__all__ = ["__version__", "compile"]
