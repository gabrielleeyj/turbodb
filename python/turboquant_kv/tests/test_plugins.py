"""Tests for plugin construction and the SafeTensors codec header (no GPU)."""

import json
import struct

import pytest

from turboquant_kv import NativeUnavailable
from turboquant_kv.config import KVConfig
from turboquant_kv.lmcache_codec import (
    TurboQuantLMCacheCodec,
    build_safetensors_header,
    turboquant_metadata,
)
from turboquant_kv.sglang_plugin import make_kv_cache
from turboquant_kv.vllm_plugin import make_backend, register, supported_dtypes


def test_vllm_register_descriptor():
    desc = register()
    assert desc["name"] == "turboquant"
    assert "turboquant3" in desc["dtypes"]
    assert callable(desc["backend_factory"])


def test_supported_dtypes():
    dtypes = supported_dtypes()
    assert dtypes == [f"turboquant{b}" for b in range(1, 9)]


def test_make_backend_and_native_guard():
    backend = make_backend("turboquant4", d_head=128)
    assert backend.get_name() == "turboquant"
    assert backend.page_size() == 16
    # Without the native library, forward() must raise NativeUnavailable.
    with pytest.raises(NativeUnavailable):
        backend.forward()


def test_make_backend_rejects_bad_dtype():
    with pytest.raises(ValueError):
        make_backend("fp8", d_head=128)


def test_sglang_native_guard():
    cache = make_kv_cache("turboquant3", d_head=64)
    with pytest.raises(NativeUnavailable):
        cache.attention()


def test_safetensors_header_framing():
    tensors = {
        "k_codes": {"dtype": "U8", "shape": [16, 96], "data_offsets": [0, 1536]},
    }
    meta = turboquant_metadata(KVConfig(d_head=128, b_key=3, b_val=3))
    blob = build_safetensors_header(tensors, meta)

    # 8-byte LE length prefix, then JSON that parses and round-trips.
    (length,) = struct.unpack("<Q", blob[:8])
    header = json.loads(blob[8 : 8 + length])
    assert header["__metadata__"]["bit_width"] == "3"
    assert header["k_codes"]["dtype"] == "U8"


def test_lmcache_codec_native_guard():
    codec = TurboQuantLMCacheCodec(KVConfig(d_head=128, b_key=3, b_val=3))
    assert codec.name == "turboquant"
    with pytest.raises(NativeUnavailable):
        codec.encode(None, None)
    with pytest.raises(NativeUnavailable):
        codec.decode(b"")
