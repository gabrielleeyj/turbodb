/**
 * turboquant_internal.h — Internal shared declarations between .cu files.
 * Not part of the public C ABI.
 */

#ifndef TURBOQUANT_INTERNAL_H
#define TURBOQUANT_INTERNAL_H

#include "turboquant.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Access the CUDA stream from a context (defined in context.cu). */
cudaStream_t tq_get_stream(tq_context_t ctx);

/* Access the device ID from a context. */
int tq_get_device_id(tq_context_t ctx);

#ifdef __cplusplus
}
#endif

/* --------------------------------------------------------------------------
 * Codebook internal structure
 * -------------------------------------------------------------------------- */

struct tq_codebook_s {
    float* centroids_d;  /* Device array of centroids, length = 2^bit_width. */
    int    bit_width;
    int    size;         /* 2^bit_width */
};

/* --------------------------------------------------------------------------
 * Rotator internal structure
 * -------------------------------------------------------------------------- */

struct tq_rotator_s {
    int      dim;        /* Original dimension */
    int      out_dim;    /* Padded to next power of 2 */
    uint64_t seed;
    float*   signs_d;    /* Device array of +1/-1 sign flips, length = out_dim. */
};

/* --------------------------------------------------------------------------
 * CUDA helpers
 * -------------------------------------------------------------------------- */

/* Check CUDA error and set context error string. Returns TQ_OK or error. */
static inline tq_status_t tq_check_cuda(tq_context_t ctx, cudaError_t err) {
    if (err == cudaSuccess) return TQ_OK;
    /* Error string is set by the caller or by context.cu's check_cuda. */
    if (err == cudaErrorMemoryAllocation) return TQ_ERR_OOM;
    return TQ_ERR_CUDA;
}

/* Ceiling division. */
static inline int tq_div_ceil(int a, int b) {
    return (a + b - 1) / b;
}

/* Next power of 2 >= n. */
static inline int tq_next_pow2(int n) {
    int v = 1;
    while (v < n) v <<= 1;
    return v;
}

/* Common block/grid sizing. */
#define TQ_BLOCK_SIZE 256

#endif /* TURBOQUANT_INTERNAL_H */
