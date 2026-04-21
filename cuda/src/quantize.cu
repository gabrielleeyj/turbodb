/**
 * quantize.cu — Scalar quantization / dequantization kernels.
 *
 * MSE quantization: rotate -> per-coord binary search on codebook -> bit-pack.
 * Codebook is loaded to shared memory (fits for b<=8: max 256 floats = 1KB).
 *
 * Bit packing: warp-cooperative for 2,3,4-bit widths. 3-bit crosses byte
 * boundaries and uses precomputed shift tables.
 *
 * Target: >= 50M vectors/sec at d=1536, b=4 on H100.
 */

#include "turboquant_internal.h"
#include <cmath>

/* --------------------------------------------------------------------------
 * Helpers
 * -------------------------------------------------------------------------- */

/**
 * Binary search for nearest centroid in shared memory codebook.
 * Returns the index of the closest centroid.
 */
__device__ int binary_search_nearest(const float* centroids, int size, float x) {
    int lo = 0, hi = size;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (centroids[mid] < x) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    /* lo is first centroid >= x. Compare lo and lo-1. */
    if (lo == 0) return 0;
    if (lo == size) return size - 1;
    float dl = x - centroids[lo - 1];
    float dr = centroids[lo] - x;
    return (dl <= dr) ? (lo - 1) : lo;
}

/**
 * Compute L2 norm of a vector segment processed by this thread.
 * Uses warp shuffle reduction.
 */
__device__ float compute_norm_partial(float val) {
    float sum = val * val;
    /* Warp-level reduction. */
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    return sum;
}

/* --------------------------------------------------------------------------
 * Quantize kernel
 *
 * One block per vector. Shared memory holds the codebook.
 * Each thread processes multiple coordinates.
 * -------------------------------------------------------------------------- */

__global__ void quantize_mse_kernel(const float* __restrict__ rotated_d,
                                     const float* __restrict__ centroids_d,
                                     int cb_size,
                                     int d, int bit_width,
                                     uint8_t* __restrict__ codes_out_d,
                                     float* __restrict__ norms_d,
                                     const float* __restrict__ input_norms_d) {
    extern __shared__ float smem_cb[];

    const int vec_idx = blockIdx.x;
    const int tid = threadIdx.x;

    /* Load codebook to shared memory. */
    for (int i = tid; i < cb_size; i += blockDim.x) {
        smem_cb[i] = centroids_d[i];
    }
    __syncthreads();

    const float* vec = rotated_d + (size_t)vec_idx * d;
    int code_bytes = tq_div_ceil(bit_width * d, 8);
    uint8_t* code_out = codes_out_d + (size_t)vec_idx * code_bytes;

    /* Quantize each coordinate: find nearest centroid index. */
    for (int i = tid; i < d; i += blockDim.x) {
        float normalized = vec[i];

        /* Binary search for nearest centroid. */
        int idx = binary_search_nearest(smem_cb, cb_size, normalized);

        /* Bit-pack: write bit_width bits at position (i * bit_width). */
        int bit_pos = i * bit_width;
        int remaining = bit_width;
        int val = idx;

        while (remaining > 0) {
            int byte_idx = bit_pos / 8;
            int bit_off = bit_pos % 8;
            int bits_avail = 8 - bit_off;
            int bits_to_write = (remaining < bits_avail) ? remaining : bits_avail;

            uint8_t chunk = (uint8_t)(val & ((1 << bits_to_write) - 1));
            atomicOr((unsigned int*)(code_out + (byte_idx & ~3)),
                     (unsigned int)(chunk << bit_off) << ((byte_idx & 3) * 8));

            val >>= bits_to_write;
            bit_pos += bits_to_write;
            remaining -= bits_to_write;
        }
    }

    /* Store input norm (passed through from pre-computation). */
    if (tid == 0 && input_norms_d != nullptr) {
        norms_d[vec_idx] = input_norms_d[vec_idx];
    }
}

/* --------------------------------------------------------------------------
 * Dequantize kernel
 * -------------------------------------------------------------------------- */

__global__ void dequantize_mse_kernel(const uint8_t* __restrict__ codes_d,
                                       const float* __restrict__ centroids_d,
                                       int cb_size,
                                       int d, int bit_width,
                                       float* __restrict__ output_d) {
    extern __shared__ float smem_cb[];

    const int vec_idx = blockIdx.x;
    const int tid = threadIdx.x;

    /* Load codebook to shared memory. */
    for (int i = tid; i < cb_size; i += blockDim.x) {
        smem_cb[i] = centroids_d[i];
    }
    __syncthreads();

    int code_bytes = tq_div_ceil(bit_width * d, 8);
    const uint8_t* code = codes_d + (size_t)vec_idx * code_bytes;
    float* vec_out = output_d + (size_t)vec_idx * d;

    /* Unpack each coordinate. */
    for (int i = tid; i < d; i += blockDim.x) {
        int bit_pos = i * bit_width;
        int remaining = bit_width;
        int val = 0;
        int bits_read = 0;

        while (remaining > 0) {
            int byte_idx = bit_pos / 8;
            int bit_off = bit_pos % 8;
            int bits_avail = 8 - bit_off;
            int bits_to_read = (remaining < bits_avail) ? remaining : bits_avail;

            int chunk = (code[byte_idx] >> bit_off) & ((1 << bits_to_read) - 1);
            val |= chunk << bits_read;

            bit_pos += bits_to_read;
            bits_read += bits_to_read;
            remaining -= bits_to_read;
        }

        val &= (1 << bit_width) - 1;
        vec_out[i] = smem_cb[val];
    }
}

/* --------------------------------------------------------------------------
 * Norm computation kernel
 * -------------------------------------------------------------------------- */

__global__ void compute_norms_kernel(const float* __restrict__ vectors,
                                      int dim, int n,
                                      float* __restrict__ norms_out) {
    int vec_idx = blockIdx.x;
    if (vec_idx >= n) return;

    const float* vec = vectors + (size_t)vec_idx * dim;
    float sum = 0.0f;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = vec[i];
        sum += v * v;
    }

    /* Block-level reduction via shared memory. */
    __shared__ float sdata[TQ_BLOCK_SIZE];
    sdata[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        norms_out[vec_idx] = sqrtf(sdata[0]);
    }
}

/**
 * Normalize vectors in-place by their norms.
 */
__global__ void normalize_kernel(float* __restrict__ vectors,
                                  const float* __restrict__ norms,
                                  int dim, int n) {
    int vec_idx = blockIdx.x;
    if (vec_idx >= n) return;

    float norm = norms[vec_idx];
    float inv_norm = (norm > 1e-30f) ? (1.0f / norm) : 0.0f;
    float* vec = vectors + (size_t)vec_idx * dim;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        vec[i] *= inv_norm;
    }
}

/**
 * Scale vectors by their stored norms (reverse of normalize).
 */
__global__ void rescale_kernel(float* __restrict__ vectors,
                                const float* __restrict__ norms,
                                int dim, int n) {
    int vec_idx = blockIdx.x;
    if (vec_idx >= n) return;

    float norm = norms[vec_idx];
    float* vec = vectors + (size_t)vec_idx * dim;

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        vec[i] *= norm;
    }
}

/* --------------------------------------------------------------------------
 * Zero-initialize codes buffer
 * -------------------------------------------------------------------------- */

__global__ void zero_codes_kernel(uint8_t* codes, size_t total_bytes) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_bytes) {
        codes[idx] = 0;
    }
}

/* --------------------------------------------------------------------------
 * Public API
 * -------------------------------------------------------------------------- */

tq_status_t tq_quantize_mse_batch(tq_context_t ctx,
                                   const float* vectors_d,
                                   int n, int dim,
                                   tq_rotator_t rot,
                                   tq_codebook_t cb,
                                   uint8_t* codes_out_d,
                                   float* norms_out_d) {
    if (ctx == nullptr) return TQ_ERR_NOT_INIT;
    if (vectors_d == nullptr || codes_out_d == nullptr || norms_out_d == nullptr)
        return TQ_ERR_INVALID_ARG;
    if (rot == nullptr || cb == nullptr) return TQ_ERR_INVALID_ARG;
    if (n <= 0 || dim < 1) return TQ_ERR_INVALID_ARG;

    cudaStream_t stream = tq_get_stream(ctx);
    int out_dim = rot->out_dim;
    int bit_width = cb->bit_width;
    int cb_size = cb->size;

    /* Allocate temporary buffers. */
    float* padded_d = nullptr;
    float* norms_d = nullptr;
    tq_status_t status;

    status = tq_device_malloc(ctx, (size_t)n * out_dim * sizeof(float),
                               (void**)&padded_d);
    if (status != TQ_OK) return status;

    status = tq_device_malloc(ctx, (size_t)n * sizeof(float),
                               (void**)&norms_d);
    if (status != TQ_OK) {
        tq_device_free(ctx, padded_d);
        return status;
    }

    /* Step 1: Compute norms. */
    compute_norms_kernel<<<n, TQ_BLOCK_SIZE, 0, stream>>>(
        vectors_d, dim, n, norms_d);

    /* Step 2: Pad vectors to out_dim (zero-padded). */
    {
        dim3 grid(tq_div_ceil(out_dim, TQ_BLOCK_SIZE), n);
        /* Reuse pad_vectors_kernel from fwht.cu — declared extern. */
        extern __global__ void pad_vectors_kernel(const float*, float*, int, int, int);
        pad_vectors_kernel<<<grid, TQ_BLOCK_SIZE, 0, stream>>>(
            vectors_d, padded_d, dim, out_dim, n);
    }

    /* Step 3: Normalize in padded space. */
    normalize_kernel<<<n, TQ_BLOCK_SIZE, 0, stream>>>(padded_d, norms_d, out_dim, n);

    /* Step 4: Apply FWHT rotation. */
    status = tq_fwht_batch(ctx, rot, padded_d, n, padded_d);
    if (status != TQ_OK) {
        tq_device_free(ctx, padded_d);
        tq_device_free(ctx, norms_d);
        return status;
    }

    /* Step 5: Zero-initialize output codes. */
    int code_bytes = tq_div_ceil(bit_width * out_dim, 8);
    size_t total_code_bytes = (size_t)n * code_bytes;
    {
        int blocks = tq_div_ceil(total_code_bytes, TQ_BLOCK_SIZE);
        zero_codes_kernel<<<blocks, TQ_BLOCK_SIZE, 0, stream>>>(
            codes_out_d, total_code_bytes);
    }

    /* Step 6: Quantize. */
    {
        size_t smem = cb_size * sizeof(float);
        quantize_mse_kernel<<<n, TQ_BLOCK_SIZE, smem, stream>>>(
            padded_d, cb->centroids_d, cb_size,
            out_dim, bit_width,
            codes_out_d, norms_out_d, norms_d);
    }

    tq_device_free(ctx, padded_d);
    tq_device_free(ctx, norms_d);

    return tq_check_cuda(ctx, cudaGetLastError());
}

tq_status_t tq_dequantize_mse_batch(tq_context_t ctx,
                                     const uint8_t* codes_d,
                                     const float* norms_d,
                                     int n, int dim,
                                     tq_rotator_t rot,
                                     tq_codebook_t cb,
                                     float* vectors_out_d) {
    if (ctx == nullptr) return TQ_ERR_NOT_INIT;
    if (codes_d == nullptr || norms_d == nullptr || vectors_out_d == nullptr)
        return TQ_ERR_INVALID_ARG;
    if (rot == nullptr || cb == nullptr) return TQ_ERR_INVALID_ARG;
    if (n <= 0 || dim < 1) return TQ_ERR_INVALID_ARG;

    cudaStream_t stream = tq_get_stream(ctx);
    int out_dim = rot->out_dim;
    int bit_width = cb->bit_width;
    int cb_size = cb->size;

    /* Allocate temp buffer for rotated-space vectors. */
    float* rotated_d = nullptr;
    tq_status_t status = tq_device_malloc(ctx, (size_t)n * out_dim * sizeof(float),
                                           (void**)&rotated_d);
    if (status != TQ_OK) return status;

    /* Step 1: Dequantize to rotated space. */
    {
        size_t smem = cb_size * sizeof(float);
        dequantize_mse_kernel<<<n, TQ_BLOCK_SIZE, smem, stream>>>(
            codes_d, cb->centroids_d, cb_size,
            out_dim, bit_width, rotated_d);
    }

    /* Step 2: Inverse rotation. */
    status = tq_fwht_inverse_batch(ctx, rot, rotated_d, n, rotated_d);
    if (status != TQ_OK) {
        tq_device_free(ctx, rotated_d);
        return status;
    }

    /* Step 3: Rescale by stored norms. */
    rescale_kernel<<<n, TQ_BLOCK_SIZE, 0, stream>>>(rotated_d, norms_d, out_dim, n);

    /* Step 4: Copy only the first dim elements to output. */
    {
        /* Truncate from out_dim to dim per vector. */
        extern __global__ void pad_vectors_kernel(const float*, float*, int, int, int);
        /* Reuse pad kernel in reverse: copy dim elements from out_dim-strided data. */
        dim3 grid(tq_div_ceil(dim, TQ_BLOCK_SIZE), n);
        /* We need a truncate kernel: */
    }

    /* Simpler: just copy dim floats per vector. Use a simple kernel. */
    {
        /* Inline truncate kernel. */
        /* For now, use cudaMemcpy2DAsync for strided copy. */
        cudaError_t err = cudaMemcpy2DAsync(
            vectors_out_d, dim * sizeof(float),
            rotated_d, out_dim * sizeof(float),
            dim * sizeof(float), n,
            cudaMemcpyDeviceToDevice, stream);
        if (err != cudaSuccess) {
            tq_device_free(ctx, rotated_d);
            return tq_check_cuda(ctx, err);
        }
    }

    tq_device_free(ctx, rotated_d);
    return tq_check_cuda(ctx, cudaGetLastError());
}

tq_status_t tq_quantize_rotate_batch(tq_context_t ctx,
                                      const float* vectors_d,
                                      int n, int dim,
                                      tq_rotator_t rot,
                                      tq_codebook_t cb,
                                      uint8_t* codes_out_d,
                                      float* norms_out_d) {
    /* Fused variant: for now, delegates to the two-step version.
     * A truly fused kernel would combine FWHT + quantize in one launch
     * for small dimensions where the entire vector fits in registers.
     * This is an optimization target for Phase 2 tuning. */
    return tq_quantize_mse_batch(ctx, vectors_d, n, dim, rot, cb,
                                  codes_out_d, norms_out_d);
}
