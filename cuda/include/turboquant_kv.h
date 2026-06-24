/**
 * turboquant_kv.h — C ABI for TurboQuant KV-cache compression.
 *
 * Extends the core TurboQuant kernels (turboquant.h) with a KV-cache-specific
 * API used by the vLLM / SGLang plugins (Component 6). Keys and values are
 * quantized per attention head; tq_kv_attention fuses dequantized-key attention
 * (softmax(Q·Kᵀ/√d)·V) into a single FlashAttention-style kernel.
 *
 * Conventions match turboquant.h: all functions return tq_status_t; pointers
 * ending in _d are device pointers the caller allocates. A tq_kv_handle_t is
 * NOT thread-safe — use one per stream.
 */
#ifndef TURBOQUANT_KV_H
#define TURBOQUANT_KV_H

#include "turboquant.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle for a KV-cache quantization context. */
typedef struct tq_kv_s* tq_kv_handle_t;

/**
 * Initialize a KV-cache context for a single attention head.
 *
 * @param d_head  Per-head dimension (e.g. 128).
 * @param b_key   Bits per key coordinate (e.g. 3 or 4). Fractional schemes use
 *                a paged mix; pass the integer floor here.
 * @param b_val   Bits per value coordinate.
 * @param out     Receives the new handle.
 * @return TQ_OK on success; TQ_ERR_UNSUPPORTED for invalid bit widths.
 */
tq_status_t tq_kv_init(int d_head, int b_key, int b_val, tq_kv_handle_t* out);

/** Destroy a KV-cache context. Safe to call with NULL. */
void tq_kv_destroy(tq_kv_handle_t h);

/**
 * Quantize a batch of key vectors. k_d points to n_tokens * d_head floats
 * (row-major); codes_out_d receives the packed codes (size given by
 * tq_kv_key_code_bytes).
 */
tq_status_t tq_kv_quantize_key(tq_kv_handle_t h, const void* k_d, int n_tokens,
                               void* codes_out_d);

/** Quantize a batch of value vectors. See tq_kv_quantize_key. */
tq_status_t tq_kv_quantize_val(tq_kv_handle_t h, const void* v_d, int n_tokens,
                               void* codes_out_d);

/**
 * Quantized-key attention: computes output = softmax(Q · K_dequant^T / sqrt(d))
 * · V_dequant, fused into a single kernel.
 *
 * @param q_d        Query vectors: n_tokens * d_head floats (the current step's
 *                   queries; for decoding this is typically 1).
 * @param k_codes_d  Packed key codes for the full context.
 * @param v_codes_d  Packed value codes for the full context.
 * @param n_tokens   Number of cached KV tokens (context length).
 * @param output_d   Receives n_tokens_q * d_head output floats.
 */
tq_status_t tq_kv_attention(tq_kv_handle_t h, const void* q_d,
                            const void* k_codes_d, const void* v_codes_d,
                            int n_tokens, void* output_d);

/** Byte size of the packed key codes for n_tokens tokens. */
size_t tq_kv_key_code_bytes(tq_kv_handle_t h, int n_tokens);

/** Byte size of the packed value codes for n_tokens tokens. */
size_t tq_kv_val_code_bytes(tq_kv_handle_t h, int n_tokens);

#ifdef __cplusplus
}
#endif

#endif /* TURBOQUANT_KV_H */
