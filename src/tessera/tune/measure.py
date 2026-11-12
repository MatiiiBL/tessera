"""Building and timing candidates.

``LocalRunner`` builds a batch of candidates in parallel processes (at most
three, so compilation does not starve the rest of the machine), then times
them one at a time in a separate measurement process, with nothing else of
the tuner's running. Every process is isolated: a candidate that crashes
(a segmentation fault in generated code), hangs past its timeout or raises
is recorded as a failure with a reason, its process is replaced, and the
run continues.

Processes are started with ``spawn`` and kept for the runner's lifetime;
each receives a task once and then only configurations, so tasks with
large constants are not sent again for every candidate.

``InlineRunner`` builds and measures in the calling process, without
isolation, for synthetic tasks and tests.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import signal
import time
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from multiprocessing.context import SpawnProcess
from types import TracebackType
from typing import Any, Literal, Protocol

from tessera.tune.task import Point, TuningTask

MAX_BUILD_WORKERS = 3

type _Op = Literal["build", "measure"]


@dataclass(frozen=True)
class MeasureResult:
    """The outcome of one candidate: seconds per call, or why it failed."""

    config: Point
    seconds: float | None
    error: str | None = None


class Runner(Protocol):
    def run(self, task: TuningTask, configs: Sequence[Point]) -> list[MeasureResult]:
        """Build and time every configuration; results are in ``configs`` order."""
        ...

    def close(self) -> None: ...


class InlineRunner:
    """Builds and measures in this process; exceptions become failures."""

    def run(self, task: TuningTask, configs: Sequence[Point]) -> list[MeasureResult]:
        results: list[MeasureResult] = []
        for config in configs:
            try:
                seconds = float(task.measure(task.build(config)))
            except Exception as e:  # a failed candidate is a result, not an error
                results.append(MeasureResult(config, None, _describe(e)))
            else:
                results.append(MeasureResult(config, seconds))
        return results

    def close(self) -> None:
        return None


class LocalRunner:
    """Parallel builds and sequential measurements in isolated processes."""

    def __init__(
        self,
        *,
        build_workers: int = MAX_BUILD_WORKERS,
        build_timeout: float = 120.0,
        measure_timeout: float = 60.0,
    ) -> None:
        if not 1 <= build_workers <= MAX_BUILD_WORKERS:
            raise ValueError(f"build_workers must be between 1 and {MAX_BUILD_WORKERS}")
        self.build_timeout = build_timeout
        self.measure_timeout = measure_timeout
        self._builders = [_Worker() for _ in range(build_workers)]
        self._measurer = _Worker()

    def run(self, task: TuningTask, configs: Sequence[Point]) -> list[MeasureResult]:
        built = _map(self._builders, task, "build", list(configs), self.build_timeout)
        results: list[MeasureResult | None] = [None] * len(configs)
        ready: list[tuple[int, Any]] = []
        for i, (ok, value) in enumerate(built):
            if ok:
                ready.append((i, value))
            else:
                results[i] = MeasureResult(configs[i], None, f"build: {value}")
        timed = _map([self._measurer], task, "measure", [v for _, v in ready], self.measure_timeout)
        for (i, _), (ok, value) in zip(ready, timed, strict=True):
            if ok:
                results[i] = MeasureResult(configs[i], float(value))
            else:
                results[i] = MeasureResult(configs[i], None, f"measure: {value}")
        return [r for r in results if r is not None]

    def close(self) -> None:
        for w in [*self._builders, self._measurer]:
            w.stop()

    def __enter__(self) -> LocalRunner:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _map(
    workers: Sequence[_Worker], task: TuningTask, op: _Op, payloads: list[Any], timeout: float
) -> list[tuple[bool, Any]]:
    """Run ``op`` on every payload, spread over ``workers``, in payload order."""
    results: list[tuple[bool, Any] | None] = [None] * len(payloads)
    pending = list(range(len(payloads)))
    busy: dict[int, tuple[_Worker, float]] = {}  # payload index -> worker, deadline
    while pending or busy:
        for w in workers:
            if not pending:
                break
            if any(bw is w for bw, _ in busy.values()):
                continue
            i = pending.pop(0)
            try:
                w.submit(task, op, i, payloads[i])
            except Exception as e:  # the task or payload cannot be sent
                w.stop()
                results[i] = (False, f"cannot send to the {op} process: {_describe(e)}")
                continue
            busy[i] = (w, time.monotonic() + timeout)
        if not busy:
            continue
        soonest = min(deadline for _, deadline in busy.values())
        conns = [w.conn for w, _ in busy.values() if w.conn is not None]
        ready = wait(conns, timeout=max(0.0, soonest - time.monotonic()))
        for i, (w, deadline) in list(busy.items()):
            conn = w.conn
            if conn is not None and conn in ready:
                del busy[i]
                try:
                    job, ok, value = conn.recv()
                except (EOFError, OSError):
                    results[i] = (False, w.death())
                    continue
                results[i] = (bool(ok), value) if job == i else (False, "protocol error")
            elif time.monotonic() >= deadline:
                del busy[i]
                w.stop()
                results[i] = (False, f"timed out after {timeout:g} s")
    return [r if r is not None else (False, "not run") for r in results]


class _Worker:
    """One child process that builds or measures on request."""

    def __init__(self) -> None:
        self.proc: SpawnProcess | None = None
        self.conn: Connection | None = None
        self.tasks: set[tuple[str, str]] = set()

    def submit(self, task: TuningTask, op: _Op, job: int, payload: Any) -> None:
        if self.proc is None or not self.proc.is_alive():
            self._start()
        assert self.conn is not None
        ident = (task.target_key, task.key)
        if ident not in self.tasks:
            self.conn.send(("task", ident, task))
            self.tasks.add(ident)
        self.conn.send(("job", job, ident, op, payload))

    def death(self) -> str:
        """Why the process ended (after its pipe closed); it is restarted on next use."""
        proc = self.proc
        self.stop()
        code = proc.exitcode if proc is not None else None
        if code is not None and code < 0:
            try:
                return f"process killed by {signal.Signals(-code).name}"
            except ValueError:
                return f"process killed by signal {-code}"
        return f"process exited with code {code}"

    def stop(self) -> None:
        if self.conn is not None:
            with contextlib.suppress(OSError, ValueError):
                self.conn.send(("stop",))
            self.conn.close()
        if self.proc is not None:
            self.proc.join(timeout=1.0)
            if self.proc.is_alive():
                self.proc.kill()
                self.proc.join(timeout=5.0)
        self.proc, self.conn, self.tasks = None, None, set()

    def _start(self) -> None:
        ctx = multiprocessing.get_context("spawn")
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=_serve, args=(child,), daemon=True, name="tessera-tune")
        proc.start()
        child.close()
        self.proc, self.conn, self.tasks = proc, parent, set()


def _serve(conn: Connection) -> None:
    """The child process: keep tasks, run jobs, answer with (job, ok, value)."""
    tasks: dict[tuple[str, str], Any] = {}
    while True:
        try:
            message = conn.recv()
        except (EOFError, OSError):
            return
        if message[0] == "stop":
            return
        if message[0] == "task":
            tasks[message[1]] = message[2]
            continue
        _, job, ident, op, payload = message
        try:
            task = tasks[ident]
            value = task.build(payload) if op == "build" else task.measure(payload)
            conn.send((job, True, value))
        except Exception as e:  # reported to the parent as this candidate's failure
            conn.send((job, False, _describe(e)))


def _describe(e: BaseException) -> str:
    text = str(e).strip().splitlines()
    return f"{type(e).__name__}: {text[0][:300]}" if text else type(e).__name__
