# TurboQuant GGUF Tensor Types

TurboDB extends the [GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
container with two custom `ggml` tensor types so quantized weights and KV-cache
snapshots can be stored in, and read back from, standard GGUF files.

These IDs live in a private high-numbered namespace so they never collide with
current or future upstream `ggml` types. Interop with upstream `llama.cpp` is
**out of scope** until/unless these types are upstreamed; the TurboDB reader and
writer round-trip them losslessly with each other.

| Type                  | ID  | Variant                         |
| --------------------- | --- | ------------------------------- |
| `GGML_TYPE_TURBOQUANT_MSE`  | 128 | MSE-optimal quantizer (`QuantMSE`)  |
| `GGML_TYPE_TURBOQUANT_PROD` | 129 | Inner-product quantizer (`QuantProd`) |

## Block layout

Tensors are stored in fixed-size blocks of **32 coordinates**. Each block is:

```
offset  size                      field
------  ------------------------  --------------------------------------------
0       2 bytes (float16)         norm        — L2 norm of the original block
2       2 bytes (uint16)          seed_offset — index into the per-model rotator table
4       ceil(32 * b / 8) bytes    codes       — packed b-bit codes, little-endian bit order
```

- `b` is the bit width (1–8), recorded in file metadata under
  `turboquant.bit_width`. It is constant for a given tensor.
- The 4-byte header (`norm` + `seed_offset`) precedes the packed codes.
- Codes are packed LSB-first: coordinate `i`'s `b` bits start at bit `i*b` of
  the body and may straddle a byte boundary (handled for all `b ≤ 8`).
- Total block size is `4 + ceil(32*b/8)` bytes, e.g. 20 bytes at `b=4`.

The rotator referenced by `seed_offset` and the codebook are reconstructed from
file metadata (`rotator_seed`, `rotator_type`, `codebook_id`), mirroring the
SafeTensors `__metadata__` schema (see `pkg/formats/safetensors`).

## Reference implementation

`pkg/formats/gguf` provides:

- `EncodeTurboQuantBlock(norm, seedOffset, codes, b)` / `DecodeTurboQuantBlock`
- `TQBlockBytes(b)` — byte size of one block at bit width `b`
- `Writer` — emits a GGUF container with these types
- `Reader` (`File`) — parses GGUF and returns raw block bytes via `Raw`
