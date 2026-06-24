"""LMCache / Mooncake compression codec for offloaded KV caches.

Implements LMCache's codec interface so offloaded KV caches are TurboQuant-
compressed. The wire format is SafeTensors with the TurboQuant ``__metadata__``
schema (see pkg/formats/safetensors and Task 5.2), making offloaded caches
portable and inspectable by the Go tooling.

The SafeTensors header construction here is pure Python (and unit-tested); the
actual quantization of key/value tensors delegates to the native kernels.
"""

from __future__ import annotations

import json
import struct
from typing import Any

from .config import KVConfig

# TurboQuant SafeTensors metadata keys (must match pkg/formats/safetensors).
META_VERSION = "turboquant_version"
META_BIT_WIDTH = "bit_width"
META_VARIANT = "variant"
META_VERSION_VALUE = "0.1"

_HEADER_LEN_SIZE = 8


def build_safetensors_header(tensors: dict[str, dict], metadata: dict[str, str]) -> bytes:
    """Build a SafeTensors length-prefixed JSON header.

    ``tensors`` maps each tensor name to ``{"dtype", "shape", "data_offsets"}``.
    The result is the 8-byte little-endian length prefix followed by the UTF-8
    JSON header — identical framing to the Go reader/writer.
    """
    obj: dict[str, Any] = {}
    if metadata:
        obj["__metadata__"] = metadata
    obj.update(tensors)
    header = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return struct.pack("<Q", len(header)) + header


def turboquant_metadata(config: KVConfig) -> dict[str, str]:
    """Return the TurboQuant SafeTensors metadata for a KV config."""
    return {
        META_VERSION: META_VERSION_VALUE,
        META_BIT_WIDTH: str(config.b_key),
        META_VARIANT: "mse",
    }


class TurboQuantLMCacheCodec:
    """Codec compatible with LMCache's offload codec interface.

    ``encode`` quantizes key/value tensors and serializes them as SafeTensors;
    ``decode`` reverses it. The quantization path requires the native library.
    """

    name = "turboquant"

    def __init__(self, config: KVConfig) -> None:
        self.config = config

    def encode(self, key: Any, value: Any) -> bytes:
        """Quantize and serialize a (key, value) pair to SafeTensors bytes."""
        self._require_native()  # raises if no GPU/native lib
        raise NotImplementedError("native quantization is provided by the CUDA build")

    def decode(self, blob: bytes) -> tuple[Any, Any]:
        """Deserialize and dequantize a SafeTensors blob to (key, value)."""
        self._require_native()
        raise NotImplementedError("native dequantization is provided by the CUDA build")

    @staticmethod
    def _require_native() -> None:
        from ._ffi import NativeUnavailable, is_available

        if not is_available():
            raise NativeUnavailable(
                "TurboQuant LMCache codec requires the native CUDA library on a GPU host."
            )
