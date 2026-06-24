"""TurboQuant KV-cache compression backend for vLLM and SGLang.

This package exposes TurboQuant as a KV-cache quantization backend. The Python
layer is a thin wrapper over the native ``libturboquant`` CUDA kernels (loaded
via :mod:`turboquant_kv._ffi`); it imports cleanly without CUDA so it can be
installed and inspected anywhere, deferring the GPU requirement to call time.
"""

from __future__ import annotations

from ._ffi import NativeUnavailable, is_available
from .config import KVConfig, is_turboquant_dtype, parse_dtype

__version__ = "0.1.0"

__all__ = [
    "KVConfig",
    "NativeUnavailable",
    "is_available",
    "is_turboquant_dtype",
    "parse_dtype",
    "__version__",
]
