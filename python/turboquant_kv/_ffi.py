"""ctypes binding to the native ``libturboquant`` KV-cache ABI.

The native library is loaded lazily and optionally: importing this module never
fails, so ``turboquant_kv`` installs and imports cleanly in a CUDA-less
environment. The actual quantize/attention calls require the compiled library
(and a GPU); they raise :class:`NativeUnavailable` when it is absent.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes.util import find_library
from typing import Optional

# tq_status_t codes mirror cuda/include/turboquant.h.
TQ_OK = 0


class NativeUnavailable(RuntimeError):
    """Raised when a native KV operation is invoked without libturboquant."""


def _candidate_names() -> list[str]:
    if sys.platform == "darwin":
        return ["libturboquant.dylib", "turboquant"]
    if sys.platform == "win32":
        return ["turboquant.dll", "turboquant"]
    return ["libturboquant.so", "turboquant"]


def _load_library() -> Optional[ctypes.CDLL]:
    # Explicit override wins.
    override = os.environ.get("TURBOQUANT_LIB")
    candidates = [override] if override else []
    candidates += _candidate_names()
    for name in candidates:
        if not name:
            continue
        try:
            return ctypes.CDLL(name)
        except OSError:
            resolved = find_library(name)
            if resolved:
                try:
                    return ctypes.CDLL(resolved)
                except OSError:
                    pass
    return None


_lib: Optional[ctypes.CDLL] = _load_library()


def is_available() -> bool:
    """Return True if the native libturboquant was loaded."""
    return _lib is not None


def _require() -> ctypes.CDLL:
    if _lib is None:
        raise NativeUnavailable(
            "libturboquant not found. Build the CUDA layer (make build-cuda) and "
            "ensure the shared library is on the loader path or set TURBOQUANT_LIB."
        )
    return _lib


def _bind() -> None:
    """Declare argument/return types for the bound symbols, if present."""
    if _lib is None:
        return
    # void* handle is represented as c_void_p; device pointers as c_void_p.
    try:
        _lib.tq_kv_init.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)
        ]
        _lib.tq_kv_init.restype = ctypes.c_int
        _lib.tq_kv_destroy.argtypes = [ctypes.c_void_p]
        _lib.tq_kv_destroy.restype = None
        _lib.tq_kv_attention.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
        ]
        _lib.tq_kv_attention.restype = ctypes.c_int
    except AttributeError:
        # Library present but missing KV symbols (older build): treat as
        # unavailable for KV operations.
        pass


_bind()


def kv_init(d_head: int, b_key: int, b_val: int) -> ctypes.c_void_p:
    """Create a native KV context handle. Raises NativeUnavailable if no lib."""
    lib = _require()
    handle = ctypes.c_void_p()
    status = lib.tq_kv_init(d_head, b_key, b_val, ctypes.byref(handle))
    if status != TQ_OK:
        raise RuntimeError(f"tq_kv_init failed with status {status}")
    return handle


def kv_destroy(handle: ctypes.c_void_p) -> None:
    if _lib is not None and handle:
        _lib.tq_kv_destroy(handle)
