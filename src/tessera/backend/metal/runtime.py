"""Python side of the Objective-C runtime shim (``shim.m``).

The shim is compiled by clang on first use into a dylib cached under
``$TESSERA_CACHE/metal`` (default ``~/.cache/tessera/metal``), named by a hash
of its source and the compiler flags, and loaded with ctypes. Every wrapper
object here owns one retained Objective-C object and releases it when it is
garbage collected.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import os
import platform
import subprocess
import tempfile
import weakref
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from tessera.dtype import DType

_CFLAGS = (
    "-fobjc-arc",
    "-O2",
    "-Wall",
    "-Wextra",
    "-dynamiclib",
    "-framework",
    "Metal",
    "-framework",
    "Foundation",
)
_ERROR_CAPACITY = 16384


class MetalError(RuntimeError):
    """The Metal runtime failed: no device, a compiler error, or a failed GPU run."""


def cache_dir() -> Path:
    """Where the compiled shim lives (``$TESSERA_CACHE/metal``)."""
    root = os.environ.get("TESSERA_CACHE") or "~/.cache/tessera"
    return Path(root).expanduser() / "metal"


def shim_source() -> str:
    """The Objective-C source of the runtime shim."""
    return resources.files(__package__).joinpath("shim.m").read_text(encoding="utf-8")


def build_shim(cc: str = "clang") -> Path:
    """Compile the shim unless a build of the same source and flags is cached."""
    source = shim_source()
    key = hashlib.sha256("\0".join((source, cc, *_CFLAGS)).encode()).hexdigest()[:16]
    out = cache_dir() / f"tessera_metal_{key}.dylib"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    # Compile into a private file and rename it, so that processes building at the
    # same time never load a half-written library.
    fd, tmp = tempfile.mkstemp(suffix=".dylib", dir=out.parent)
    os.close(fd)
    with tempfile.TemporaryDirectory() as work:
        src = Path(work) / "shim.m"
        src.write_text(source, encoding="utf-8")
        result = subprocess.run(
            [cc, *_CFLAGS, str(src), "-o", tmp], capture_output=True, text=True, check=False
        )
    if result.returncode != 0:
        os.unlink(tmp)
        raise MetalError(f"compiling the Metal shim failed:\n{result.stderr}")
    os.replace(tmp, out)
    return out


class _DeviceInfo(ctypes.Structure):
    _fields_ = (
        ("name", ctypes.c_char * 128),
        ("max_threads_per_threadgroup", ctypes.c_uint64),
        ("max_threadgroup_memory", ctypes.c_uint64),
        ("max_buffer_length", ctypes.c_uint64),
        ("recommended_working_set", ctypes.c_uint64),
        ("has_unified_memory", ctypes.c_int32),
        ("has_stage_timestamps", ctypes.c_int32),
    )


class _PipelineInfo(ctypes.Structure):
    _fields_ = (
        ("max_total_threads", ctypes.c_uint64),
        ("thread_execution_width", ctypes.c_uint64),
        ("static_threadgroup_memory", ctypes.c_uint64),
    )


_P = ctypes.c_void_p
_SIGNATURES: dict[str, tuple[Any, list[Any]]] = {
    "tm_release": (None, [_P]),
    "tm_context_create": (_P, []),
    "tm_context_info": (None, [_P, ctypes.POINTER(_DeviceInfo)]),
    "tm_library_create": (_P, [_P, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]),
    "tm_pipeline_create": (
        _P,
        [_P, _P, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t],
    ),
    "tm_pipeline_info_get": (None, [_P, ctypes.POINTER(_PipelineInfo)]),
    "tm_buffer_create": (_P, [_P, ctypes.c_uint64]),
    "tm_buffer_contents": (_P, [_P]),
    "tm_buffer_length": (ctypes.c_uint64, [_P]),
    "tm_program_create": (_P, [_P]),
    "tm_program_add": (
        ctypes.c_int,
        [
            _P,
            _P,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
            ctypes.POINTER(_P),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_int32,
        ],
    ),
    "tm_program_size": (ctypes.c_uint32, [_P]),
    "tm_program_run": (
        ctypes.c_int,
        [
            _P,
            ctypes.c_uint32,
            ctypes.c_int32,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ],
    ),
    "tm_program_profile": (
        ctypes.c_int,
        [_P, ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_char_p, ctypes.c_size_t],
    ),
}


@functools.cache
def _lib() -> ctypes.CDLL:
    if platform.system() != "Darwin":
        raise MetalError("Metal needs macOS")
    lib = ctypes.CDLL(str(build_shim()))
    for name, (restype, argtypes) in _SIGNATURES.items():
        fn = getattr(lib, name)
        fn.restype = restype
        fn.argtypes = argtypes
    return lib


def _error_buffer() -> ctypes.Array[ctypes.c_char]:
    return ctypes.create_string_buffer(_ERROR_CAPACITY)


class _Handle:
    """Owns one retained Objective-C object."""

    def __init__(self, handle: int | None, what: str) -> None:
        if not handle:
            raise MetalError(f"could not create {what}")
        self.handle = handle
        weakref.finalize(self, _lib().tm_release, handle)


@dataclass(frozen=True)
class DeviceInfo:
    name: str
    max_threads_per_threadgroup: int
    max_threadgroup_memory: int
    max_buffer_length: int
    recommended_working_set: int
    has_unified_memory: bool
    has_stage_timestamps: bool


class Device(_Handle):
    """The GPU with its command queue."""

    def __init__(self) -> None:
        super().__init__(_lib().tm_context_create(), "a Metal device")
        raw = _DeviceInfo()
        _lib().tm_context_info(self.handle, ctypes.byref(raw))
        self.info = DeviceInfo(
            name=raw.name.decode(),
            max_threads_per_threadgroup=raw.max_threads_per_threadgroup,
            max_threadgroup_memory=raw.max_threadgroup_memory,
            max_buffer_length=raw.max_buffer_length,
            recommended_working_set=raw.recommended_working_set,
            has_unified_memory=bool(raw.has_unified_memory),
            has_stage_timestamps=bool(raw.has_stage_timestamps),
        )

    def compile(self, source: str) -> Library:
        """Compile MSL source; raises ``MetalError`` with the compiler's messages."""
        err = _error_buffer()
        handle = _lib().tm_library_create(self.handle, source.encode(), err, _ERROR_CAPACITY)
        if not handle:
            raise MetalError(f"MSL compilation failed:\n{err.value.decode(errors='replace')}")
        return Library(self, handle)

    def buffer(self, nbytes: int) -> Buffer:
        """A shared-storage buffer of ``nbytes`` (zero-filled)."""
        return Buffer(self, nbytes)

    def program(self) -> Program:
        """An empty launch list."""
        return Program(self)


class Library(_Handle):
    """A compiled MSL library."""

    def __init__(self, device: Device, handle: int) -> None:
        super().__init__(handle, "a Metal library")
        self.device = device

    def pipeline(self, name: str) -> Pipeline:
        """The compute pipeline of kernel ``name``."""
        err = _error_buffer()
        handle = _lib().tm_pipeline_create(
            self.device.handle, self.handle, name.encode(), err, _ERROR_CAPACITY
        )
        if not handle:
            raise MetalError(f"pipeline {name}: {err.value.decode(errors='replace')}")
        return Pipeline(self, name, handle)


class Pipeline(_Handle):
    """One kernel ready to launch, with the limits the compiler gave it."""

    def __init__(self, library: Library, name: str, handle: int) -> None:
        super().__init__(handle, f"pipeline {name}")
        self.library = library
        self.name = name
        raw = _PipelineInfo()
        _lib().tm_pipeline_info_get(handle, ctypes.byref(raw))
        self.max_total_threads = int(raw.max_total_threads)
        self.thread_execution_width = int(raw.thread_execution_width)
        self.static_threadgroup_memory = int(raw.static_threadgroup_memory)


class Buffer(_Handle):
    """A buffer in unified memory: ``view`` gives NumPy arrays over its bytes."""

    def __init__(self, device: Device, nbytes: int) -> None:
        super().__init__(_lib().tm_buffer_create(device.handle, nbytes), f"a {nbytes}-byte buffer")
        self.nbytes = nbytes
        self._address = int(_lib().tm_buffer_contents(self.handle))

    def view(
        self, dtype: DType | npt.DTypeLike, shape: Sequence[int], offset: int = 0
    ) -> npt.NDArray[Any]:
        """A NumPy array over ``shape`` elements starting ``offset`` bytes in; it
        keeps the buffer alive."""
        np_dtype = dtype.numpy if isinstance(dtype, DType) else np.dtype(dtype)
        count = int(np.prod(shape, dtype=np.int64))
        nbytes = count * np_dtype.itemsize
        if offset < 0 or offset + nbytes > self.nbytes:
            raise ValueError(f"view of {nbytes} bytes at {offset} exceeds {self.nbytes} bytes")
        raw = (ctypes.c_char * max(nbytes, 1)).from_address(self._address + offset)
        array = np.frombuffer(_Keep(raw, self), dtype=np_dtype, count=count)
        return array.reshape(tuple(shape))


class _Keep:
    """Exposes a ctypes array's memory while holding the buffer that owns it."""

    def __init__(self, raw: ctypes.Array[ctypes.c_char], owner: Buffer) -> None:
        self.raw = raw
        self.owner = owner

    def __buffer__(self, flags: int) -> memoryview:
        return memoryview(self.raw).cast("B")


@dataclass(frozen=True)
class RunTimes:
    wall: list[float]  # seconds from encoding to completion, per run
    gpu: list[float]  # seconds the GPU spent on the command buffer, per run


class Program(_Handle):
    """A launch list: runs encode every launch into one command buffer."""

    def __init__(self, device: Device) -> None:
        super().__init__(_lib().tm_program_create(device.handle), "a program")
        self.device = device
        self.labels: list[str] = []
        self._keep: list[object] = []

    def add(
        self,
        pipeline: Pipeline,
        grid: Sequence[int],
        threads: Sequence[int],
        args: Sequence[tuple[Buffer, int]],
        *,
        barrier: bool = True,
        label: str | None = None,
    ) -> None:
        """Append a launch of ``grid`` threadgroups of ``threads`` threads binding
        ``args`` (buffer, byte offset) to buffer indices 0, 1, .... ``barrier``
        makes it wait for every earlier launch."""
        if len(grid) != 3 or len(threads) != 3:
            raise ValueError("grid and threads have three dimensions")
        if prod3(threads) > pipeline.max_total_threads:
            raise MetalError(
                f"{pipeline.name}: {prod3(threads)} threads per threadgroup exceed the "
                f"pipeline's limit of {pipeline.max_total_threads}"
            )
        grid_c = (ctypes.c_uint32 * 3)(*grid)
        threads_c = (ctypes.c_uint32 * 3)(*threads)
        buffers = (_P * max(len(args), 1))(*(b.handle for b, _ in args))
        offsets = (ctypes.c_uint64 * max(len(args), 1))(*(off for _, off in args))
        name = label or pipeline.name
        status = _lib().tm_program_add(
            self.handle,
            pipeline.handle,
            name.encode(),
            grid_c,
            threads_c,
            len(args),
            buffers,
            offsets,
            int(barrier),
        )
        if status < 0:
            raise MetalError("out of memory adding a launch")
        self.labels.append(name)
        self._keep.extend([pipeline, *(b for b, _ in args)])

    def __len__(self) -> int:
        return len(self.labels)

    def run(self, repeat: int = 1, *, serial: bool = False, timeout: float = 0.0) -> RunTimes:
        """Run the program ``repeat`` times, one command buffer per run.
        ``serial`` lets Metal order every launch instead of using the barriers;
        ``timeout`` 0 waits as long as the GPU takes."""
        wall = (ctypes.c_double * max(repeat, 1))()
        gpu = (ctypes.c_double * max(repeat, 1))()
        err = _error_buffer()
        status = _lib().tm_program_run(
            self.handle, repeat, int(serial), timeout, wall, gpu, err, _ERROR_CAPACITY
        )
        if status != 0:
            raise MetalError(f"GPU run failed: {err.value.decode(errors='replace')}")
        return RunTimes(list(wall)[:repeat], list(gpu)[:repeat])

    def profile(self, *, timeout: float = 0.0) -> list[float]:
        """GPU seconds of each launch, from timestamps the GPU takes around it."""
        out = (ctypes.c_double * max(len(self), 1))()
        err = _error_buffer()
        status = _lib().tm_program_profile(self.handle, timeout, out, err, _ERROR_CAPACITY)
        if status != 0:
            raise MetalError(f"GPU profile failed: {err.value.decode(errors='replace')}")
        return list(out)[: len(self)]


def prod3(values: Sequence[int]) -> int:
    return int(values[0]) * int(values[1]) * int(values[2])


@functools.cache
def device() -> Device:
    """The system's default GPU (created once); raises ``MetalError`` without one."""
    return Device()


@functools.cache
def available() -> bool:
    """Whether a Metal device can be used here (macOS, clang, a GPU)."""
    if platform.system() != "Darwin":
        return False
    try:
        device()
    except (MetalError, OSError):
        return False
    return True
