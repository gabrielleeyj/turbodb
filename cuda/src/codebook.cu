/**
 * codebook.cu — Codebook and Rotator lifecycle on the GPU.
 */

#include "turboquant_internal.h"
#include <cstdlib>
#include <cmath>

/* --------------------------------------------------------------------------
 * Codebook
 * -------------------------------------------------------------------------- */

tq_status_t tq_codebook_create(tq_context_t ctx,
                                const float* centroids_h,
                                int bit_width,
                                tq_codebook_t* out) {
    if (ctx == nullptr) return TQ_ERR_NOT_INIT;
    if (centroids_h == nullptr || out == nullptr) return TQ_ERR_INVALID_ARG;
    if (bit_width < 1 || bit_width > 8) return TQ_ERR_INVALID_ARG;

    int size = 1 << bit_width;

    tq_codebook_t cb = (tq_codebook_t)calloc(1, sizeof(struct tq_codebook_s));
    if (cb == nullptr) return TQ_ERR_OOM;

    cb->bit_width = bit_width;
    cb->size = size;

    cudaStream_t stream = tq_get_stream(ctx);
    cudaError_t err = cudaMalloc(&cb->centroids_d, size * sizeof(float));
    if (err != cudaSuccess) {
        free(cb);
        return TQ_ERR_OOM;
    }

    err = cudaMemcpyAsync(cb->centroids_d, centroids_h,
                           size * sizeof(float),
                           cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(cb->centroids_d);
        free(cb);
        return tq_check_cuda(ctx, err);
    }

    /* Ensure copy completes before returning. */
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        cudaFree(cb->centroids_d);
        free(cb);
        return tq_check_cuda(ctx, err);
    }

    *out = cb;
    return TQ_OK;
}

void tq_codebook_destroy(tq_codebook_t cb) {
    if (cb == nullptr) return;
    cudaFree(cb->centroids_d);
    free(cb);
}

int tq_codebook_bit_width(tq_codebook_t cb) {
    return cb ? cb->bit_width : 0;
}

int tq_codebook_size(tq_codebook_t cb) {
    return cb ? cb->size : 0;
}

/* --------------------------------------------------------------------------
 * Rotator
 * -------------------------------------------------------------------------- */

/**
 * Kernel to generate sign flips from a seed.
 * Each element is +1.0f or -1.0f based on a simple hash of (seed, index).
 */
__global__ void generate_signs_kernel(float* signs, int n, uint64_t seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    /* Simple hash: mix seed with index to get a deterministic sign. */
    uint64_t h = seed ^ ((uint64_t)idx * 0x9e3779b97f4a7c15ULL);
    h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
    h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
    h = h ^ (h >> 31);

    signs[idx] = (h & 1) ? 1.0f : -1.0f;
}

tq_status_t tq_rotator_create(tq_context_t ctx,
                                int dim, uint64_t seed,
                                tq_rotator_t* out) {
    if (ctx == nullptr) return TQ_ERR_NOT_INIT;
    if (out == nullptr || dim < 1) return TQ_ERR_INVALID_ARG;

    int out_dim = tq_next_pow2(dim);

    tq_rotator_t rot = (tq_rotator_t)calloc(1, sizeof(struct tq_rotator_s));
    if (rot == nullptr) return TQ_ERR_OOM;

    rot->dim = dim;
    rot->out_dim = out_dim;
    rot->seed = seed;

    cudaStream_t stream = tq_get_stream(ctx);

    cudaError_t err = cudaMalloc(&rot->signs_d, out_dim * sizeof(float));
    if (err != cudaSuccess) {
        free(rot);
        return TQ_ERR_OOM;
    }

    /* Generate sign flips on GPU. */
    int blocks = tq_div_ceil(out_dim, TQ_BLOCK_SIZE);
    generate_signs_kernel<<<blocks, TQ_BLOCK_SIZE, 0, stream>>>(
        rot->signs_d, out_dim, seed);

    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        cudaFree(rot->signs_d);
        free(rot);
        return tq_check_cuda(ctx, err);
    }

    *out = rot;
    return TQ_OK;
}

void tq_rotator_destroy(tq_rotator_t rot) {
    if (rot == nullptr) return;
    cudaFree(rot->signs_d);
    free(rot);
}

int tq_rotator_dim(tq_rotator_t rot) {
    return rot ? rot->dim : 0;
}

int tq_rotator_out_dim(tq_rotator_t rot) {
    return rot ? rot->out_dim : 0;
}
