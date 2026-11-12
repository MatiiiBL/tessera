"""Building and timing candidates.

``LocalRunner`` builds a batch of candidates in parallel processes
(``build_workers``, 3 by default), then times them one at a time in a
separate measurement process, with nothing else of the tuner's running.
The two phases never overlap, so the build parallelism does not affect the
measurements; it only bounds how much CPU and memory the clang processes
of one batch take at once. Every process is isolated: a candidate that crashes
(a segmentation fault in generated code), hangs past its timeout or raises
is recorded as a failure with a reason, its process is replaced, and the
run continues.

Each process is a fresh interpreter running this module
(``python -m tessera.tune.measure``) that talks to the tuner over two pipes.
Unlike ``multiprocessing``'s ``spawn``, it never re-imports the caller's
main script, so the runner works from any script. Tasks and results cross
the pipes pickled: a task must be an instance of a class its module
defines at top level. Processes are kept for the runner's lifetime and
receive each task once, then only configurations, so a task with large
constants is not sent again for every candidate.

``InlineRunner`` builds and measures in the calling process, without
isolation, for synthetic tasks and tests.
"""

from __future__ import annotations

import contextlib
import io
import os
import pickle
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from types import TracebackType
from typing import Any, Literal, Protocol

from tessera.tune.task import Point, TuningTask

DEFAULT_BUILD_WORKERS = 3

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
        build_workers: int = DEFAULT_BUILD_WORKERS,
        build_timeout: float = 120.0,
        measure_timeout: float = 60.0,
    ) -> None:
        if build_workers < 1:
            raise ValueError(f"build_workers must be at least 1, got {build_workers}")
        self.build_timeout = build_timeout
        self.measure_timeout = measure_timeout
        self._builders = [_Worker() for _ in range(build_workers)]
        self._measurer = _Worker()
        self.build_seconds = 0.0  # wall-clock time spent in build phases
        self.measure_seconds = 0.0  # wall-clock time spent in measurement phases

    def run(self, task: TuningTask, configs: Sequence[Point]) -> list[MeasureResult]:
        started = time.monotonic()
        built = _map(self._builders, task, "build", list(configs), self.build_timeout)
        self.build_seconds += time.monotonic() - started
        results: list[MeasureResult | None] = [None] * len(configs)
        ready: list[tuple[int, Any]] = []
        for i, (ok, value) in enumerate(built):
            if ok:
                ready.append((i, value))
            else:
                results[i] = MeasureResult(configs[i], None, f"build: {value}")
        started = time.monotonic()
        timed = _map([self._measurer], task, "measure", [v for _, v in ready], self.measure_timeout)
        self.measure_seconds += time.monotonic() - started
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
        self.proc: subprocess.Popen[bytes] | None = None
        self.conn: Connection | None = None  # answers from the child
        self._jobs: Connection | None = None  # requests to the child
        self.tasks: set[tuple[str, str]] = set()

    def submit(self, task: TuningTask, op: _Op, job: int, payload: Any) -> None:
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        assert self._jobs is not None
        ident = (task.target_key, task.key)
        if ident not in self.tasks:
            self._jobs.send(("task", ident, task))
            self.tasks.add(ident)
        self._jobs.send(("job", job, ident, op, payload))

    def death(self) -> str:
        """Why the process ended (after its pipe closed); it is restarted on next use."""
        proc = self.proc
        code = None
        if proc is not None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                code = proc.wait(timeout=5.0)
        self.stop()
        if code is not None and code < 0:
            try:
                return f"process killed by {signal.Signals(-code).name}"
            except ValueError:
                return f"process killed by signal {-code}"
        return f"process exited with code {code}"

    def stop(self) -> None:
        if self._jobs is not None:
            with contextlib.suppress(OSError, ValueError):
                self._jobs.send(("stop",))
            self._jobs.close()
        if self.conn is not None:
            self.conn.close()
        if self.proc is not None:
            try:
                self.proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5.0)
        self.proc, self.conn, self._jobs, self.tasks = None, None, None, set()

    def _start(self) -> None:
        # A fresh interpreter running this module, rather than multiprocessing's
        # spawn, which re-runs the caller's main script in every child.
        jobs_read, jobs_write = os.pipe()
        answers_read, answers_write = os.pipe()
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
        self.proc = subprocess.Popen(
            [sys.executable, "-m", __spec__.name, str(jobs_read), str(answers_write)],
            pass_fds=(jobs_read, answers_write),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            env=env,
        )
        os.close(jobs_read)
        os.close(answers_write)
        self._jobs = Connection(jobs_write, readable=False)
        self.conn = Connection(answers_read, writable=False)
        self.tasks = set()


def _serve(jobs: Connection, answers: Connection) -> None:
    """The child process: keep tasks, run jobs, answer with (job, ok, value)."""
    # Ctrl-C reaches the whole process group; the parent decides what to stop.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    tasks: dict[tuple[str, str], Any] = {}
    broken: dict[tuple[str, str], str] = {}
    while True:
        try:
            data = jobs.recv_bytes()
        except (EOFError, OSError):
            return
        try:
            message = pickle.loads(data)
        except Exception as e:  # a task whose class this process cannot import
            ident = _task_ident(data)
            if ident is not None:
                broken[ident] = f"cannot load the task here: {_describe(e)}"
            continue
        if message[0] == "stop":
            return
        if message[0] == "task":
            tasks[message[1]] = message[2]
            continue
        _, job, ident, op, payload = message
        try:
            if ident in broken:
                raise RuntimeError(broken[ident])
            task = tasks[ident]
            value = task.build(payload) if op == "build" else task.measure(payload)
            answers.send((job, True, value))
        except Exception as e:  # reported to the parent as this candidate's failure
            answers.send((job, False, _describe(e)))


def _task_ident(data: bytes) -> tuple[str, str] | None:
    """The identity of a task message that failed to unpickle, read without the task."""

    class _Stub:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __setstate__(self, state: object) -> None:
            pass

    class _Lenient(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            try:
                return super().find_class(module, name)
            except (ImportError, AttributeError):
                return _Stub

    try:
        message = _Lenient(io.BytesIO(data)).load()
    except Exception:
        return None
    if isinstance(message, tuple) and len(message) == 3 and message[0] == "task":
        ident: tuple[str, str] = message[1]
        return ident
    return None


def _describe(e: BaseException) -> str:
    text = str(e).strip().splitlines()
    return f"{type(e).__name__}: {text[0][:300]}" if text else type(e).__name__


if __name__ == "__main__":
    _serve(
        Connection(int(sys.argv[1]), writable=False), Connection(int(sys.argv[2]), readable=False)
    )
