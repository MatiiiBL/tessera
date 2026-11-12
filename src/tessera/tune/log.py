"""Tuning logs: every measurement, one JSON object per line.

```json
{"target": "c-sme-t4-Apple_M4", "task": "matmul:3f2a...", "config": {"tile_m": [8, 16]},
 "seconds": 0.000412, "error": null, "time": "2026-11-12T10:00:00Z"}
```

Logs are append-only: a record is written (and flushed) as soon as its
measurement finishes, so an interrupted run loses nothing, and a restarted
run reads the log and skips what it already measured. Lines that do not
parse (a run killed in the middle of a write) are ignored. ``compile()``
looks up the fastest record of each (target, task) pair; a configuration
that the current template no longer accepts (its knobs changed) is skipped
in favour of the next fastest.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from tessera.tune.space import KnobValue, config_from_json, config_json


@dataclass(frozen=True)
class Record:
    """One measurement of one configuration of one task on one target."""

    target: str
    task: str
    config: Mapping[str, KnobValue]
    seconds: float | None
    error: str | None = None
    time: str = ""

    @property
    def ok(self) -> bool:
        return self.seconds is not None and self.error is None

    def to_json(self) -> str:
        return json.dumps(
            {
                "target": self.target,
                "task": self.task,
                "config": json.loads(config_json(self.config)),
                "seconds": self.seconds,
                "error": self.error,
                "time": self.time or now(),
            }
        )

    @classmethod
    def from_json(cls, line: str) -> Record:
        raw = json.loads(line)
        seconds = raw.get("seconds")
        return cls(
            target=str(raw["target"]),
            task=str(raw["task"]),
            config=config_from_json(json.dumps(raw["config"])),
            seconds=float(seconds) if seconds is not None else None,
            error=raw.get("error"),
            time=str(raw.get("time", "")),
        )


def now() -> str:
    """The current UTC time in the log format."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class TuningLog:
    """A JSON-lines log file (or an in-memory log when ``path`` is None)."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._records: list[Record] = []
        self._offset = 0  # bytes of the file already parsed

    def append(self, record: Record) -> None:
        """Write one record and flush it to disk."""
        if record.time == "":
            record = replace(record, time=now())
        if self.path is None:
            self._records.append(record)
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(record.to_json() + "\n")
            f.flush()
            os.fsync(f.fileno())

    def records(self) -> list[Record]:
        """Every record in file order (reads only what was appended since the last call)."""
        if self.path is None or not self.path.exists():
            return list(self._records)
        with self.path.open("rb") as f:
            f.seek(self._offset)
            data = f.read()
        end = data.rfind(b"\n") + 1  # a line without its newline is still being written
        self._offset += end
        self._records.extend(_parse(data[:end].decode("utf-8", errors="replace").splitlines()))
        return list(self._records)

    def measured(self, target: str, task: str) -> dict[str, Record]:
        """The latest record of each configuration of (target, task), by ``config_json``."""
        out: dict[str, Record] = {}
        for r in self.records():
            if r.target == target and r.task == task:
                out[config_json(r.config)] = r
        return out


def _parse(lines: Iterable[str]) -> Iterator[Record]:
    for line in lines:
        if not line.strip():
            continue
        try:
            yield Record.from_json(line)
        except (ValueError, KeyError, TypeError):
            continue


class BestConfigs:
    """The fastest logged configurations, for ``compile()``.

    Built from any number of logs (files, or directories of ``*.jsonl``
    files). Only successful measurements count.
    """

    def __init__(self, records: Iterable[Record] = ()) -> None:
        self._by_task: dict[tuple[str, str], list[Record]] = {}
        for r in records:
            if r.ok:
                self._by_task.setdefault((r.target, r.task), []).append(r)
        for rs in self._by_task.values():
            rs.sort(key=lambda r: r.seconds if r.seconds is not None else float("inf"))

    @classmethod
    def load(cls, *paths: str | os.PathLike[str]) -> BestConfigs:
        """Read log files and directories of ``*.jsonl`` logs (missing paths are skipped)."""
        records: list[Record] = []
        for path in map(Path, paths):
            files = sorted(path.glob("**/*.jsonl")) if path.is_dir() else [path]
            for file in files:
                if file.exists():
                    records.extend(TuningLog(file).records())
        return cls(records)

    def __len__(self) -> int:
        return len(self._by_task)

    def targets(self) -> set[str]:
        """The target keys that have at least one successful record."""
        return {target for target, _ in self._by_task}

    def lookup(
        self,
        target: str,
        task: str,
        accepts: Callable[[Mapping[str, KnobValue]], bool] | None = None,
    ) -> Record | None:
        """The fastest record of (target, task) whose config ``accepts`` allows."""
        for r in self._by_task.get((target, task), []):
            if accepts is None or accepts(r.config):
                return r
        return None
