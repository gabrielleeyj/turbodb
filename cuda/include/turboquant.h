/**
 * turboquant.h — C ABI for TurboQuant CUDA kernel layer.
 *
 * All functions return tq_status_t. Pointers ending in _d are device pointers;
 * callers manage allocation. One tq_context_t per CUDA stream.
 *
 * Thread safety: a tq_context_t is NOT thread-safe. Use one per thread/stream.
 * The underlying CUDA stream is owned by the context.
 */

#ifndef TURBOQUANT_H
#define TURBOQUANT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------------
 * Status codes
 * -------------------------------------------------------------------------- */

typedef enum {
    TQ_OK              = 0,
    TQ_ERR_CUDA        = 1,   /* CUDA runtime error (check tq_last_error) */
    TQ_ERR_OOM         = 2,   /* device or host out of memory */
    TQ_ERR_INVALID_ARG = 3,   /* invalid argument value */
    TQ_ERR_DIM_NOT_POW2 = 4,  /* dimension must be power of 2 for FWHT */
    TQ_ERR_UNSUPPORTED = 5,   /* unsupported bit-width or configuration */
    TQ_ERR_NOT_INIT    = 6,   /* context not initialized */
    TQ_ERR_INTERNAL    = 7    /* unexpected internal error */
} tq_status_t;

/* --------------------------------------------------------------------------
 * Opaque handles
 * -------------------------------------------------------------------------- */

typedef struct tq_context_s*  tq_context_t;
typedef struct tq_codebook_s* tq_codebook_t;
typedef struct tq_rotator_s*  tq_rotator_t;

/* --------------------------------------------------------------------------
 * Context lifecycle
 * -------------------------------------------------------------------------- */

/**
 * Initialize a TurboQuant context on the given CUDA device.
 * Creates a dedicated CUDA stream for all operations on this context.
 *
 * @param device_id  CUDA device ordinal (0-based).
 * @param out        Receives the new context handle.
 * @return TQ_OK on success.
 */
tq_status_t tq_init(int device_id, tq_context_t* out);

/**
 * Destroy a context and free all associated resources.
 * Safe to call with NULL. After this call, the handle is invalid.
 */
void tq_destroy(tq_context_t ctx);

/**
 * Synchronize the context's CUDA stream. Blocks until all enqueued
 * operations complete.
 */
tq_status_t tq_sync(tq_context_t ctx);

/**
 * Return a human-readable description of the last CUDA error.
 * The returned string is valid until the next call on this context.
 */
const char* tq_last_error(tq_context_t ctx);

/* --------------------------------------------------------------------------
 * Device memory helpers
 * -------------------------------------------------------------------------- */

/**
 * Allocate device memory.
 * @param ctx       Context (determines device).
 * @param size      Bytes to allocate.
 * @param ptr_out   Receives device pointer.
 */
tq_status_t tq_device_malloc(tq_context_t ctx, size_t size, void** ptr_out);

/**
 * Free device memory previously allocated with tq_device_malloc.
 */
tq_status_t tq_device_free(tq_context_t ctx, void* ptr_d);

/**
 * Copy host memory to device (async on context's stream).
 */
tq_status_t tq_memcpy_h2d(tq_context_t ctx, void* dst_d,
                           const void* src_h, size_t size);

/**
 * Copy device memory to host (async on context's stream).
 */
tq_status_t tq_memcpy_d2h(tq_context_t ctx, void* dst_h,
                           const void* src_d, size_t size);

/* --------------------------------------------------------------------------
 * Codebook management
 * -------------------------------------------------------------------------- */

/**
 * Create a codebook on the device from host centroids.
 *
 * @param ctx         Context.
 * @param centroids_h Host array of centroids (float32), length = 2^bit_width.
 * @param bit_width   Bits per coordinate (1..8).
 * @param out         Receives the codebook handle.
 */
tq_status_t tq_codebook_create(tq_context_t ctx,
                                const float* centroids_h,
                                int bit_width,
                                tq_codebook_t* out);

/**
 * Destroy a codebook and free device memory.
 */
void tq_codebook_destroy(tq_codebook_t cb);

/**
 * Return the bit-width of a codebook.
 */
int tq_codebook_bit_width(tq_codebook_t cb);

/**
 * Return the number of centroids (2^bit_width).
 */
int tq_codebook_size(tq_codebook_t cb);

/* --------------------------------------------------------------------------
 * Rotator management
 * -------------------------------------------------------------------------- */

/**
 * Create a Hadamard rotator for the given dimension.
 * The dimension is padded to the next power of 2 internally.
 *
 * @param ctx   Context.
 * @param dim   Original vector dimension.
 * @param seed  Deterministic seed for sign flips.
 * @param out   Receives the rotator handle.
 */
tq_status_t tq_rotator_create(tq_context_t ctx,
                                int dim, uint64_t seed,
                                tq_rotator_t* out);

/**
 * Destroy a rotator and free device memory.
 */
void tq_rotator_destroy(tq_rotator_t rot);

/**
 * Return the original (unpadded) input dimension.
 */
int tq_rotator_dim(tq_rotator_t rot);

/**
 * Return the padded output dimension (power of 2).
 */
int tq_rotator_out_dim(tq_rotator_t rot);

/* --------------------------------------------------------------------------
 * FWHT — Fast Walsh-Hadamard Transform (Task 2.2)
 * -------------------------------------------------------------------------- */

/**
 * Apply batched randomized Hadamard rotation (sign-flip + FWHT + sign-flip).
 *
 * @param ctx         Context.
 * @param rot         Rotator handle.
 * @param vectors_d   Device input:  n x out_dim row-major float32.
 * @param n           Number of vectors in the batch.
 * @param output_d    Device output: n x out_dim row-major float32.
 *                    May alias vectors_d for in-place operation.
 */
tq_status_t tq_fwht_batch(tq_context_t ctx,
                           tq_rotator_t rot,
                           const float* vectors_d,
                           int n,
                           float* output_d);

/**
 * Apply batched inverse Hadamard rotation (sign-flip + FWHT + sign-flip,
 * then scale by 1/out_dim).
 */
tq_status_t tq_fwht_inverse_batch(tq_context_t ctx,
                                   tq_rotator_t rot,
                                   const float* vectors_d,
                                   int n,
                                   float* output_d);

/* --------------------------------------------------------------------------
 * Quantize / Dequantize (Task 2.3)
 * -------------------------------------------------------------------------- */

/**
 * Batch MSE quantization: rotate -> per-coord binary search -> bit-pack.
 *
 * @param ctx         Context.
 * @param vectors_d   Device input: n x dim float32 (original, unpadded dim).
 * @param n           Number of vectors.
 * @param dim         Original vector dimension (before padding).
 * @param rot         Rotator handle.
 * @param cb          Codebook handle.
 * @param codes_out_d Device output: packed bit-stream, n x ceil(bit_width * out_dim / 8) bytes.
 * @param norms_out_d Device output: per-vector L2 norms, n x float32.
 */
tq_status_t tq_quantize_mse_batch(tq_context_t ctx,
                                   const float* vectors_d,
                                   int n, int dim,
                                   tq_rotator_t rot,
                                   tq_codebook_t cb,
                                   uint8_t* codes_out_d,
                                   float* norms_out_d);

/**
 * Batch MSE dequantization: unpack -> centroid lookup -> inverse rotation.
 *
 * @param ctx         Context.
 * @param codes_d     Device input: packed codes, same layout as quantize output.
 * @param norms_d     Device input: per-vector norms.
 * @param n           Number of vectors.
 * @param dim         Original vector dimension.
 * @param rot         Rotator handle.
 * @param cb          Codebook handle.
 * @param vectors_out_d Device output: n x dim float32.
 */
tq_status_t tq_dequantize_mse_batch(tq_context_t ctx,
                                     const uint8_t* codes_d,
                                     const float* norms_d,
                                     int n, int dim,
                                     tq_rotator_t rot,
                                     tq_codebook_t cb,
                                     float* vectors_out_d);

/**
 * Fused rotate-then-quantize: combines rotation and quantization in a
 * single kernel launch for vectors that fit in registers.
 * Same parameters as tq_quantize_mse_batch.
 */
tq_status_t tq_quantize_rotate_batch(tq_context_t ctx,
                                      const float* vectors_d,
                                      int n, int dim,
                                      tq_rotator_t rot,
                                      tq_codebook_t cb,
                                      uint8_t* codes_out_d,
                                      float* norms_out_d);

/* --------------------------------------------------------------------------
 * QJL — Quantized Johnson-Lindenstrauss (Task 2.4)
 * -------------------------------------------------------------------------- */

/**
 * Compute batched QJL 1-bit sketches.
 *
 * @param ctx         Context.
 * @param vectors_d   Device input: n x dim float32.
 * @param n           Number of vectors.
 * @param dim         Vector dimension.
 * @param proj_dim    Number of random projections (sign bits per vector).
 * @param seed        Deterministic seed for Gaussian matrix generation.
 * @param signs_out_d Device output: packed sign bits, n x ceil(proj_dim/8) bytes.
 * @param norms_out_d Device output: per-vector L2 norms, n x float32.
 */
tq_status_t tq_qjl_sketch_batch(tq_context_t ctx,
                                  const float* vectors_d,
                                  int n, int dim,
                                  int proj_dim,
                                  uint64_t seed,
                                  uint8_t* signs_out_d,
                                  float* norms_out_d);

/**
 * Estimate inner products using QJL sketches.
 *
 * @param ctx           Context.
 * @param query_d       Device input: single query vector, dim float32.
 * @param signs_d       Device input: packed sign bits for n database vectors.
 * @param norms_d       Device input: norms for n database vectors.
 * @param n             Number of database vectors.
 * @param dim           Vector dimension.
 * @param proj_dim      Number of projection dimensions.
 * @param seed          Same seed used for sketching.
 * @param results_out_d Device output: n estimated inner products, float32.
 */
tq_status_t tq_qjl_estimate_ip_batch(tq_context_t ctx,
                                       const float* query_d,
                                       const uint8_t* signs_d,
                                       const float* norms_d,
                                       int n, int dim,
                                       int proj_dim,
                                       uint64_t seed,
                                       float* results_out_d);

/* --------------------------------------------------------------------------
 * Search (Task 2.5)
 * -------------------------------------------------------------------------- */

/**
 * Result of a top-K search: vector ID and score (inner product estimate).
 */
typedef struct {
    int64_t id;
    float   score;
} tq_search_result_t;

/**
 * Brute-force inner-product search over quantized database codes.
 * Computes the TurboQuant unbiased IP estimator (Theorem 2) for each
 * query-database pair, then returns top-K results per query.
 *
 * @param ctx           Context.
 * @param query_codes_d Device input: quantized query codes (MSE part).
 * @param query_norms_d Device input: query norms.
 * @param query_signs_d Device input: query QJL sign bits (NULL if MSE-only).
 * @param query_res_norms_d Device input: query residual norms (NULL if MSE-only).
 * @param n_queries     Number of queries.
 * @param db_codes_d    Device input: database quantized codes.
 * @param db_norms_d    Device input: database norms.
 * @param db_signs_d    Device input: database QJL sign bits (NULL if MSE-only).
 * @param db_res_norms_d Device input: database residual norms (NULL if MSE-only).
 * @param n_db          Number of database vectors.
 * @param dim           Original vector dimension.
 * @param bit_width     Quantization bit-width.
 * @param proj_dim      QJL projection dimension (0 if MSE-only).
 * @param cb            Codebook handle.
 * @param k             Number of top results to return per query.
 * @param results_out_h Host output: n_queries x k results, sorted by score descending.
 */
tq_status_t tq_search_brute_force(tq_context_t ctx,
                                   const uint8_t* query_codes_d,
                                   const float* query_norms_d,
                                   const uint8_t* query_signs_d,
                                   const float* query_res_norms_d,
                                   int n_queries,
                                   const uint8_t* db_codes_d,
                                   const float* db_norms_d,
                                   const uint8_t* db_signs_d,
                                   const float* db_res_norms_d,
                                   int n_db,
                                   int dim, int bit_width,
                                   int proj_dim,
                                   tq_codebook_t cb,
                                   int k,
                                   tq_search_result_t* results_out_h);

/* --------------------------------------------------------------------------
 * Device info
 * -------------------------------------------------------------------------- */

/**
 * Query GPU memory usage on the context's device.
 * @param free_bytes  Receives free device memory in bytes.
 * @param total_bytes Receives total device memory in bytes.
 */
tq_status_t tq_device_memory_info(tq_context_t ctx,
                                   size_t* free_bytes,
                                   size_t* total_bytes);

/**
 * Return the compute capability (major * 10 + minor) of the context's device.
 * E.g., 80 for A100 (SM 8.0), 90 for H100 (SM 9.0).
 */
tq_status_t tq_device_compute_capability(tq_context_t ctx, int* cc_out);

#ifdef __cplusplus
}
#endif

#endif /* TURBOQUANT_H */
