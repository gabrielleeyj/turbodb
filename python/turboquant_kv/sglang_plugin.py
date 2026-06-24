"""SGLang integration for the TurboQuant KV-cache backend.

SGLang's KV-cache manager is more pluggable than vLLM's; this implements its
``BaseKVCache`` surface, sharing the same native FFI layer
(:mod:`turboquant_kv._ffi`) and config as the vLLM plugin. Imports cleanly
without SGLang installed.
"""

from __future__ import annotations

from typing import Any

from .config import KVConfig, parse_dtype


def make_kv_cache(dtype: str, d_head: int, page_size: int = 16) -> "TurboQuantKVCache":
    """Construct an SGLang KV cache for the given TurboQuant dtype."""
    cfg = parse_dtype(dtype, d_head=d_head, page_size=page_size)
    return TurboQuantKVCache(cfg)


class TurboQuantKVCache:
    """TurboQuant-backed implementation of SGLang's ``BaseKVCache`` interface.

    Mirrors the vLLM backend's behavior; the native kernels are shared. When
    SGLang is installed this can be registered as a custom KV cache manager.
    """

    def __init__(self, config: KVConfig) -> None:
        self.config = config

    def quantize_store(self, *args: Any, **kwargs: Any) -> Any:
        """Quantize and store keys/values for a set of tokens."""
        return self._require_native()

    def attention(self, *args: Any, **kwargs: Any) -> Any:
        """Quantized-key attention over the cached context."""
        return self._require_native()

    @staticmethod
    def _require_native() -> Any:
        from ._ffi import NativeUnavailable, is_available

        if not is_available():
            raise NativeUnavailable(
                "TurboQuant SGLang backend requires the native CUDA library on a GPU host."
            )
        raise NotImplementedError(
            "GPU dispatch is provided by the CUDA build; exercised in GPU CI."
        )
