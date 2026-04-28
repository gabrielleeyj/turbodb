/**
 * fwht.cu — Fast Walsh-Hadamard Transform kernels for TurboQuant.
 *
 * Implements batched FWHT with fused sign multiplication for the randomized
 * Hadamard rotation used in TurboQuant. The algorithm applies:
 *   output = S2 * H * S1 * input
 * where S1, S2 are diagonal sign matrices and H is the Hadamard matrix.
 *
 * For the inverse:
 *   output = S1 * H * S2 * input * (1/d)
 * since H is its own inverse up to scaling.
 *
 * Strategy:
 * - Small d (<=1024): single-kernel shared-memory butterfly.
 * - Large d (>1024): multi-pass with shared memory tiles.
 *
 * Target: >= 1.1 TB/s effective throughput on H100 for d=4096, N=1M.
 */

#include "turboquant_internal.h"
#include <cstdio>

/* --------------------------------------------------------------------------
 * Shared-memory bank-conflict avoidance.
 *
 * 4-byte shared memory has 32 banks on all CUDA GPUs since CC 2.0. The FWHT
 * butterfly at stage `half` reads smem[i] and smem[i + half] simultaneously
 * across a warp; whenever `half` is a multiple of 32 (32, 64, 128, ...) the
 * two halves of the pair land in the same bank, producing 2-way bank
 * conflicts that double the load latency.
 *
 * Standard fix: pad the index by `i / 32`, i.e. add one slot every 32
 * elements. This shifts banks so that adjacent stride-32 accesses land in
 * different banks. The smem footprint grows by ~3.1% — well within budget.
 * -------------------------------------------------------------------------- */
#define FWHT_PAD(i) ((i) + ((i) >> 5))
#define FWHT_PADDED_SIZE(d) ((d) + ((d) >> 5))

/* --------------------------------------------------------------------------
 * Shared-memory FWHT butterfly kernel
 *
 * Each block processes one vector. d must be a power of 2 and fit in
 * shared memory (d <= 4096 with ~17KB shared mem at float32 after padding).
 * -------------------------------------------------------------------------- */

__global__ void fwht_forward_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     const float* __restrict__ signs,
                                     int d) {
    extern __shared__ float smem[];

    const int vec_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const float* vec_in = input + (size_t)vec_idx * d;
    float* vec_out = output + (size_t)vec_idx * d;

    /* Load into shared memory with first sign flip (S1). */
    for (int i = tid; i < d; i += blockDim.x) {
        smem[FWHT_PAD(i)] = vec_in[i] * signs[i];
    }
    __syncthreads();

    /* Butterfly stages: log2(d) stages.
     * Each thread handles non-overlapping butterfly pairs to avoid
     * concurrent read/write of the same shared-memory locations. */
    for (int half = 1; half < d; half <<= 1) {
        int pairs = d / 2;
        for (int p = tid; p < pairs; p += blockDim.x) {
            /* Map pair index to the two butterfly indices.
             * Within each block of (2*half) elements, the first half
             * indices pair with the second half. */
            int block_size = half << 1;
            int block_id = p / half;
            int offset = p % half;
            int i = block_id * block_size + offset;
            int j = i + half;

            float a = smem[FWHT_PAD(i)];
            float b = smem[FWHT_PAD(j)];
            smem[FWHT_PAD(i)] = a + b;
            smem[FWHT_PAD(j)] = a - b;
        }
        __syncthreads();
    }

    /* Apply second sign flip (S2) and write output. */
    for (int i = tid; i < d; i += blockDim.x) {
        vec_out[i] = smem[FWHT_PAD(i)] * signs[i];
    }
}

__global__ void fwht_inverse_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     const float* __restrict__ signs,
                                     int d) {
    extern __shared__ float smem[];

    const int vec_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const float* vec_in = input + (size_t)vec_idx * d;
    float* vec_out = output + (size_t)vec_idx * d;

    /* For inverse: apply S2 first, then FWHT, then S1, then scale 1/d. */
    const float scale = 1.0f / (float)d;

    /* Load with S2 sign flip. */
    for (int i = tid; i < d; i += blockDim.x) {
        smem[FWHT_PAD(i)] = vec_in[i] * signs[i];
    }
    __syncthreads();

    /* Butterfly stages (same as forward — H is symmetric).
     * Each thread handles non-overlapping butterfly pairs. */
    for (int half = 1; half < d; half <<= 1) {
        int pairs = d / 2;
        for (int p = tid; p < pairs; p += blockDim.x) {
            int block_size = half << 1;
            int block_id = p / half;
            int offset = p % half;
            int i = block_id * block_size + offset;
            int j = i + half;

            float a = smem[FWHT_PAD(i)];
            float b = smem[FWHT_PAD(j)];
            smem[FWHT_PAD(i)] = a + b;
            smem[FWHT_PAD(j)] = a - b;
        }
        __syncthreads();
    }

    /* Apply S1 sign flip and 1/d scaling. */
    for (int i = tid; i < d; i += blockDim.x) {
        vec_out[i] = smem[FWHT_PAD(i)] * signs[i] * scale;
    }
}

/* --------------------------------------------------------------------------
 * Large-d FWHT (d > shared memory capacity)
 *
 * Uses a tiled approach: each butterfly stage that spans within a tile is
 * done in shared memory; cross-tile stages are done in global memory.
 * -------------------------------------------------------------------------- */

#define FWHT_TILE_SIZE 1024

__global__ void fwht_global_butterfly_kernel(float* __restrict__ data,
                                              int d, int half) {
    /* Each thread handles one butterfly pair for this stage. */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = d / 2;
    if (idx >= total_pairs) return;

    /* Compute the pair indices for this butterfly stage. */
    int block_size = half << 1;
    int block_id = idx / half;
    int offset = idx % half;
    int i = block_id * block_size + offset;
    int j = i + half;

    /* Read from the single vector at blockIdx.y offset. */
    size_t base = (size_t)blockIdx.y * d;
    float a = data[base + i];
    float b = data[base + j];
    data[base + i] = a + b;
    data[base + j] = a - b;
}

__global__ void apply_signs_kernel(float* __restrict__ data,
                                    const float* __restrict__ signs,
                                    int d) {
    int vec_idx = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;

    size_t base = (size_t)vec_idx * d;
    data[base + i] *= signs[i];
}

__global__ void apply_signs_and_scale_kernel(float* __restrict__ data,
                                              const float* __restrict__ signs,
                                              int d, float scale) {
    int vec_idx = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;

    size_t base = (size_t)vec_idx * d;
    data[base + i] *= signs[i] * scale;
}

/* --------------------------------------------------------------------------
 * Pad input: copy dim-sized vectors into out_dim-sized vectors (zero pad).
 * -------------------------------------------------------------------------- */

__global__ void pad_vectors_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int dim, int out_dim, int n) {
    int vec_idx = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx >= n || i >= out_dim) return;

    size_t in_base = (size_t)vec_idx * dim;
    size_t out_base = (size_t)vec_idx * out_dim;

    output[out_base + i] = (i < dim) ? input[in_base + i] : 0.0f;
}

/* --------------------------------------------------------------------------
 * Public API
 * -------------------------------------------------------------------------- */

/* Maximum shared memory per block for the butterfly kernel (in floats). */
static const int MAX_SMEM_FLOATS = 4096;  /* 16 KB */

tq_status_t tq_fwht_batch(tq_context_t ctx,
                           tq_rotator_t rot,
                           const float* vectors_d,
                           int n,
                           float* output_d) {
    if (ctx == nullptr) return TQ_ERR_NOT_INIT;
    if (rot == nullptr || vectors_d == nullptr || output_d == nullptr)
        return TQ_ERR_INVALID_ARG;
    if (n <= 0) return TQ_ERR_INVALID_ARG;

    int d = rot->out_dim;
    cudaStream_t stream = tq_get_stream(ctx);

    if (d <= MAX_SMEM_FLOATS) {
        /* Small-d path: one block per vector, butterfly in shared memory. */
        int threads = (d < TQ_BLOCK_SIZE) ? d : TQ_BLOCK_SIZE;
        size_t smem_bytes = (size_t)FWHT_PADDED_SIZE(d) * sizeof(float);

        fwht_forward_kernel<<<n, threads, smem_bytes, stream>>>(
            vectors_d, output_d, rot->signs_d, d);
        TQ_CHECK_LAUNCH(ctx);
    } else {
        /* Large-d path: copy to output, apply signs, global butterfly. */
        if (output_d != vectors_d) {
            cudaError_t err = cudaMemcpyAsync(
                output_d, vectors_d, (size_t)n * d * sizeof(float),
                cudaMemcpyDeviceToDevice, stream);
            if (err != cudaSuccess) return tq_check_cuda(ctx, err);
        }

        /* Apply S1 sign flip. */
        dim3 sign_grid(tq_div_ceil(d, TQ_BLOCK_SIZE), n);
        apply_signs_kernel<<<sign_grid, TQ_BLOCK_SIZE, 0, stream>>>(
            output_d, rot->signs_d, d);
        TQ_CHECK_LAUNCH(ctx);

        /* Butterfly stages. */
        int total_pairs = d / 2;
        for (int half = 1; half < d; half <<= 1) {
            dim3 grid(tq_div_ceil(total_pairs, TQ_BLOCK_SIZE), n);
            fwht_global_butterfly_kernel<<<grid, TQ_BLOCK_SIZE, 0, stream>>>(
                output_d, d, half);
            TQ_CHECK_LAUNCH(ctx);
        }

        /* Apply S2 sign flip. */
        apply_signs_kernel<<<sign_grid, TQ_BLOCK_SIZE, 0, stream>>>(
            output_d, rot->signs_d, d);
    }

    return tq_check_cuda(ctx, cudaGetLastError());
}

tq_status_t tq_fwht_inverse_batch(tq_context_t ctx,
                                   tq_rotator_t rot,
                                   const float* vectors_d,
                                   int n,
                                   float* output_d) {
    if (ctx == nullptr) return TQ_ERR_NOT_INIT;
    if (rot == nullptr || vectors_d == nullptr || output_d == nullptr)
        return TQ_ERR_INVALID_ARG;
    if (n <= 0) return TQ_ERR_INVALID_ARG;

    int d = rot->out_dim;
    cudaStream_t stream = tq_get_stream(ctx);
    float scale = 1.0f / (float)d;

    if (d <= MAX_SMEM_FLOATS) {
        int threads = (d < TQ_BLOCK_SIZE) ? d : TQ_BLOCK_SIZE;
        size_t smem_bytes = (size_t)FWHT_PADDED_SIZE(d) * sizeof(float);

        fwht_inverse_kernel<<<n, threads, smem_bytes, stream>>>(
            vectors_d, output_d, rot->signs_d, d);
        TQ_CHECK_LAUNCH(ctx);
    } else {
        if (output_d != vectors_d) {
            cudaError_t err = cudaMemcpyAsync(
                output_d, vectors_d, (size_t)n * d * sizeof(float),
                cudaMemcpyDeviceToDevice, stream);
            if (err != cudaSuccess) return tq_check_cuda(ctx, err);
        }

        dim3 sign_grid(tq_div_ceil(d, TQ_BLOCK_SIZE), n);

        /* Apply S2 first for inverse. */
        apply_signs_kernel<<<sign_grid, TQ_BLOCK_SIZE, 0, stream>>>(
            output_d, rot->signs_d, d);
        TQ_CHECK_LAUNCH(ctx);

        /* Butterfly stages. */
        int total_pairs = d / 2;
        for (int half = 1; half < d; half <<= 1) {
            dim3 grid(tq_div_ceil(total_pairs, TQ_BLOCK_SIZE), n);
            fwht_global_butterfly_kernel<<<grid, TQ_BLOCK_SIZE, 0, stream>>>(
                output_d, d, half);
            TQ_CHECK_LAUNCH(ctx);
        }

        /* Apply S1 and scale. */
        apply_signs_and_scale_kernel<<<sign_grid, TQ_BLOCK_SIZE, 0, stream>>>(
            output_d, rot->signs_d, d, scale);
    }

    return tq_check_cuda(ctx, cudaGetLastError());
}
