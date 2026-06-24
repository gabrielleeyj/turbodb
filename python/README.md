# turboquant-kv

TurboQuant as a KV-cache compression backend for vLLM and SGLang (Component 6).

The Python package is a thin wrapper over the native `libturboquant` CUDA
kernels. It **installs and imports cleanly without CUDA** — the GPU requirement
is deferred to call time, so the package can be inspected, configured, and
unit-tested anywhere.

## Install

```sh
pip install turboquant-kv            # core, no GPU needed to import
pip install "turboquant-kv[vllm]"    # + vLLM integration
pip install "turboquant-kv[sglang]"  # + SGLang integration
```

## Use with vLLM

```python
from vllm import LLM
llm = LLM(model="...", kv_cache_dtype="turboquant3")
```

`kv_cache_dtype` is `turboquant<bits>` for `bits` in 1..8.

## Layout

| Module | Task | Purpose |
| --- | --- | --- |
| `config.py` | — | `KVConfig`, dtype parsing |
| `_ffi.py` | 6.1 | ctypes binding to `libturboquant` (lazy, optional) |
| `vllm_plugin.py` | 6.2 | vLLM `kv_cache_dtype` backend + entry point |
| `sglang_plugin.py` | 6.3 | SGLang `BaseKVCache` implementation |
| `lmcache_codec.py` | 6.4 | LMCache offload codec (SafeTensors wire format) |
| `tests/` | 6.5 | config/plugin tests + GPU-gated accuracy/throughput harness |

The C ABI consumed via `_ffi.py` is declared in `cuda/include/turboquant_kv.h`.

## Status

- Tasks 6.1–6.4: Python surface, config, FFI loader, and plugin/codec skeletons
  are implemented and unit-tested without a GPU (`pytest` → 20+ passing).
- The fused quantized-key attention kernel (`tq_kv_attention`) and the
  accuracy/throughput validation (Task 6.5) require the CUDA build on a GPU
  host; those paths raise `NativeUnavailable` off-GPU and are exercised in GPU
  CI (see `tests/test_validation.py`, gated by `TURBOQUANT_RUN_GPU_TESTS=1`).
