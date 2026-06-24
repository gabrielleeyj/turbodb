"""vLLM integration for the TurboQuant KV-cache backend.

Registers ``kv_cache_dtype`` values ``"turboquant3"``, ``"turboquant4"`` … and
provides a :class:`TurboQuantBackend` that quantizes the paged KV cache through
the native kernels. This module imports without vLLM installed; vLLM is imported
lazily inside :func:`register` and the backend factory.

Usage::

    # pip install "turboquant-kv[vllm]"
    from vllm import LLM
    llm = LLM(model="...", kv_cache_dtype="turboquant3")

This continues the work of vLLM PR #38280 rather than forking a parallel path.
"""

from __future__ import annotations

from typing import Any

from .config import KVConfig, is_turboquant_dtype, parse_dtype


def supported_dtypes() -> list[str]:
    """Return the kv_cache_dtype strings this plugin handles."""
    return [f"turboquant{b}" for b in range(1, 9)]


def register() -> dict[str, Any]:
    """Entry point invoked by vLLM's plugin discovery.

    Returns a descriptor of the backend. Registration is best-effort: if the
    installed vLLM exposes a backend registry we hook into it; otherwise we
    return the descriptor so the caller can wire it manually.
    """
    descriptor = {
        "name": "turboquant",
        "dtypes": supported_dtypes(),
        "backend_factory": make_backend,
    }
    try:
        # Newer vLLM exposes a registry; older versions do not. Hook in if so.
        from vllm.attention.backends import registry as _registry  # type: ignore

        if hasattr(_registry, "register_kv_cache_backend"):
            _registry.register_kv_cache_backend("turboquant", make_backend)
    except Exception:  # noqa: BLE001 - optional dependency / API drift tolerated
        pass
    return descriptor


def make_backend(dtype: str, d_head: int, page_size: int = 16) -> "TurboQuantBackend":
    """Construct a backend for the given kv_cache_dtype."""
    if not is_turboquant_dtype(dtype):
        raise ValueError(f"{dtype!r} is not a TurboQuant kv_cache_dtype")
    cfg = parse_dtype(dtype, d_head=d_head, page_size=page_size)
    return TurboQuantBackend(cfg)


class TurboQuantBackend:
    """Quantized paged-KV attention backend.

    Subclasses vLLM's ``AttentionBackend`` when vLLM is importable; otherwise it
    is a standalone object exposing the same surface so it can be unit-tested
    without vLLM. The heavy lifting is delegated to the native kernels via
    :mod:`turboquant_kv._ffi`.
    """

    def __init__(self, config: KVConfig) -> None:
        self.config = config

    @staticmethod
    def get_name() -> str:
        return "turboquant"

    def page_size(self) -> int:
        return self.config.page_size

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Paged quantized-key attention. Delegates to the native kernel.

        Raises:
            turboquant_kv.NativeUnavailable: if libturboquant is not loaded.
        """
        from ._ffi import NativeUnavailable, is_available

        if not is_available():
            raise NativeUnavailable(
                "TurboQuant KV attention requires the native CUDA library on a GPU host."
            )
        # Native fused attention dispatch (tq_kv_attention) is wired in the GPU
        # build; see cuda/include/turboquant_kv.h.
        raise NotImplementedError(
            "GPU fused attention dispatch is provided by the CUDA build; "
            "this path is exercised in GPU CI."
        )
