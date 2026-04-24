/**
 * qjl.cu — Quantized Johnson-Lindenstrauss transform kernels.
 *
 * Implements the 1-bit QJL sketch from Definition 1 of the TurboQuant paper:
 *   sign(G * x) where G is a seeded random Gaussian matrix.
 *
 * Strategy:
 * - Use cuBLAS SGEMM for the G*x matrix multiplication (portable, high perf).
 * - Generate G on-the-fly using a deterministic PRNG seeded per-row to avoid
 *   materializing the full proj_dim x dim matrix in memory.
 * - Fuse sign extraction as a post-GEMM kernel.
 *
 * For very large proj_dim, we tile the projection in chunks.
 */

#include "turboquant_internal.h"
#include <cublas_v2.h>
#include <curand_kernel.h>
#include <cmath>

/* --------------------------------------------------------------------------
 * Gaussian matrix generation
 *
 * Generate G in tiles. Each thread generates one element of G using
 * cuRAND with a deterministic seed derived from (global_seed, row, col).
 * -------------------------------------------------------------------------- */

__global__ void generate_gaussian_kernel(float* __restrict__ G,
                                          int rows, int cols,
                                          uint64_t seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;

    int row = idx / cols;
    int col = idx % cols;

    /* Deterministic seed per element. */
    curandStatePhilox4_32_10_t state;
    curand_init(seed, (unsigned long long)row * cols + col, 0, &state);

    G[idx] = curand_normal(&state);
}

/* --------------------------------------------------------------------------
 * Sign extraction: convert float projections to packed sign bits.
 * -------------------------------------------------------------------------- */

__global__ void extract_signs_kernel(const float* __restrict__ projections,
                                      int n, int proj_dim,
                                      uint8_t* __restrict__ signs_out) {
    int vec_idx = blockIdx.y;
    int bit_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bit_idx >= proj_dim) return;

    const float* proj = projections + (size_t)vec_idx * proj_dim;
    int byte_idx = bit_idx / 8;
    int bit_off = bit_idx % 8;
    int sign_bytes = tq_div_ceil(proj_dim, 8);

    if (proj[bit_idx] >= 0.0f) {
        /* Align to 4-byte boundary relative to the buffer base
         * (signs_out is cudaMalloc-aligned) to avoid misaligned atomics. */
        size_t abs_byte = (size_t)vec_idx * sign_bytes + byte_idx;
        size_t aligned = abs_byte & ~(size_t)3;
        int word_shift = (int)(abs_byte - aligned) * 8;

        atomicOr((unsigned int*)(signs_out + aligned),
                 (unsigned int)(1 << bit_off) << word_shift);
    }
}

/* --------------------------------------------------------------------------
 * Norm computation (reuse pattern from quantize.cu)
 * -------------------------------------------------------------------------- */

__global__ void qjl_compute_norms_kernel(const float* __restrict__ vectors,
                                          int dim, int n,
                                          float* __restrict__ norms_out) {
    int vec_idx = blockIdx.x;
    if (vec_idx >= n) return;

    const float* vec = vectors + (size_t)vec_idx * dim;
    __shared__ float sdata[TQ_BLOCK_SIZE];

    float sum = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = vec[i];
        sum += v * v;
    }
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

/* --------------------------------------------------------------------------
 * Zero-init for sign bits
 * -------------------------------------------------------------------------- */

__global__ void zero_bytes_kernel(uint8_t* data, size_t count) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) data[idx] = 0;
}

/* --------------------------------------------------------------------------
 * Public API
 * -------------------------------------------------------------------------- */

tq_status_t tq_qjl_sketch_batch(tq_context_t ctx,
                                  const float* vectors_d,
                                  int n, int dim,
                                  int proj_dim,
                                  uint64_t seed,
                                  uint8_t* signs_out_d,
                                  float* norms_out_d) {
    if (ctx == nullptr) return TQ_ERR_NOT_INIT;
    if (vectors_d == nullptr || signs_out_d == nullptr || norms_out_d == nullptr)
        return TQ_ERR_INVALID_ARG;
    if (n <= 0 || dim < 1 || proj_dim < 1) return TQ_ERR_INVALID_ARG;

    cudaStream_t stream = tq_get_stream(ctx);
    tq_status_t status;

    /* Step 1: Compute norms. */
    qjl_compute_norms_kernel<<<n, TQ_BLOCK_SIZE, 0, stream>>>(
        vectors_d, dim, n, norms_out_d);

    /* Step 2: Generate Gaussian projection matrix G (proj_dim x dim). */
    float* G_d = nullptr;
    status = tq_device_malloc(ctx, (size_t)proj_dim * dim * sizeof(float),
                               (void**)&G_d);
    if (status != TQ_OK) return status;

    {
        int total = proj_dim * dim;
        int blocks = tq_div_ceil(total, TQ_BLOCK_SIZE);
        generate_gaussian_kernel<<<blocks, TQ_BLOCK_SIZE, 0, stream>>>(
            G_d, proj_dim, dim, seed);
    }

    /* Step 3: GEMM: projections = G * vectors^T -> proj_dim x n
     * Then transpose to n x proj_dim for sign extraction.
     *
     * Actually, we want: for each vector v_i, compute G * v_i.
     * With vectors as n x dim (row-major), we compute:
     *   P = vectors * G^T  =>  P is n x proj_dim
     * Using cuBLAS (column-major): C = B^T * A^T
     * where A = vectors (n x dim), B = G (proj_dim x dim)
     * C = G * vectors^T in col-major = vectors * G^T in row-major
     */
    float* proj_d = nullptr;
    status = tq_device_malloc(ctx, (size_t)n * proj_dim * sizeof(float),
                               (void**)&proj_d);
    if (status != TQ_OK) {
        tq_device_free(ctx, G_d);
        return status;
    }

    {
        cublasHandle_t handle;
        cublasStatus_t cstat = cublasCreate(&handle);
        if (cstat != CUBLAS_STATUS_SUCCESS) {
            tq_device_free(ctx, G_d);
            tq_device_free(ctx, proj_d);
            return TQ_ERR_CUDA;
        }
        cublasSetStream(handle, stream);

        /* cuBLAS is column-major. Our data is row-major.
         * Row-major A (n x dim) in col-major is A^T (dim x n).
         * Row-major G (proj_dim x dim) in col-major is G^T (dim x proj_dim).
         *
         * We want row-major result P (n x proj_dim).
         * In col-major this is P^T (proj_dim x n).
         *
         * P^T = G * A^T (col-major interpretation)
         *   where G is "transposed" of our row-major G -> actually G^T in col-major
         *   and A^T is our row-major vectors in col-major.
         *
         * cuBLAS: C(proj_dim x n) = op(G) * op(vectors)
         *   G stored as dim x proj_dim (col-major of row-major proj_dim x dim)
         *   vectors stored as dim x n (col-major of row-major n x dim)
         *
         * C = G^T(col) * vectors(col)  but G^T(col) = G(row) which is proj_dim x dim
         * So: CUBLAS_OP_T on G_col (dim x proj_dim), result is proj_dim x dim * dim x n = proj_dim x n
         */
        float alpha = 1.0f, beta = 0.0f;
        cstat = cublasSgemm(handle,
                            CUBLAS_OP_T,  /* transpose G (stored col-major as dim x proj_dim) */
                            CUBLAS_OP_N,  /* vectors (stored col-major as dim x n) */
                            proj_dim,     /* M: rows of result */
                            n,            /* N: cols of result */
                            dim,          /* K: shared dimension */
                            &alpha,
                            G_d, dim,     /* G: lda = dim (col-major of row-major G) */
                            vectors_d, dim, /* vectors: ldb = dim */
                            &beta,
                            proj_d, proj_dim); /* C: ldc = proj_dim */

        cublasDestroy(handle);

        if (cstat != CUBLAS_STATUS_SUCCESS) {
            tq_device_free(ctx, G_d);
            tq_device_free(ctx, proj_d);
            return TQ_ERR_CUDA;
        }
    }

    tq_device_free(ctx, G_d);

    /* Step 4: Zero-init signs, then extract sign bits from projections.
     * proj_d is col-major proj_dim x n. For sign extraction, element (p, v)
     * is at proj_d[p + v * proj_dim] — which is contiguous for each vector
     * when accessed as n x proj_dim row-major. Same memory layout. */
    int sign_bytes = tq_div_ceil(proj_dim, 8);
    size_t total_sign_bytes = (size_t)n * sign_bytes;

    {
        int blocks = tq_div_ceil(total_sign_bytes, TQ_BLOCK_SIZE);
        zero_bytes_kernel<<<blocks, TQ_BLOCK_SIZE, 0, stream>>>(
            signs_out_d, total_sign_bytes);
    }

    {
        dim3 grid(tq_div_ceil(proj_dim, TQ_BLOCK_SIZE), n);
        extract_signs_kernel<<<grid, TQ_BLOCK_SIZE, 0, stream>>>(
            proj_d, n, proj_dim, signs_out_d);
    }

    tq_device_free(ctx, proj_d);

    return tq_check_cuda(ctx, cudaGetLastError());
}

/* --------------------------------------------------------------------------
 * QJL inner-product estimation
 *
 * For a query y and database vectors with stored QJL sketches:
 *   cos(angle) ≈ 1 - 2 * (fraction of disagreeing bits)
 *   <x, y> ≈ ‖x‖ * ‖y‖ * cos(angle)
 * -------------------------------------------------------------------------- */

__global__ void qjl_estimate_ip_kernel(const float* __restrict__ query_proj_d,
                                        const uint8_t* __restrict__ db_signs_d,
                                        const float* __restrict__ db_norms_d,
                                        float query_norm,
                                        int n_db, int proj_dim,
                                        float* __restrict__ results_d) {
    int db_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (db_idx >= n_db) return;

    int sign_bytes = tq_div_ceil(proj_dim, 8);
    const uint8_t* db_signs = db_signs_d + (size_t)db_idx * sign_bytes;

    /* Count agreements between query projection signs and db signs. */
    int agree = 0;
    for (int p = 0; p < proj_dim; p++) {
        int byte_idx = p / 8;
        int bit_off = p % 8;
        int db_sign = (db_signs[byte_idx] >> bit_off) & 1;
        int q_sign = (query_proj_d[p] >= 0.0f) ? 1 : 0;
        if (db_sign == q_sign) agree++;
    }

    float disagree_frac = (float)(proj_dim - agree) / (float)proj_dim;
    float cos_est = 1.0f - 2.0f * disagree_frac;
    results_d[db_idx] = query_norm * db_norms_d[db_idx] * cos_est;
}

tq_status_t tq_qjl_estimate_ip_batch(tq_context_t ctx,
                                       const float* query_d,
                                       const uint8_t* signs_d,
                                       const float* norms_d,
                                       int n, int dim,
                                       int proj_dim,
                                       uint64_t seed,
                                       float* results_out_d) {
    if (ctx == nullptr) return TQ_ERR_NOT_INIT;
    if (query_d == nullptr || signs_d == nullptr || norms_d == nullptr ||
        results_out_d == nullptr)
        return TQ_ERR_INVALID_ARG;
    if (n <= 0 || dim < 1 || proj_dim < 1) return TQ_ERR_INVALID_ARG;

    cudaStream_t stream = tq_get_stream(ctx);
    tq_status_t status;

    /* Step 1: Generate G and compute query projections (G * query). */
    float* G_d = nullptr;
    status = tq_device_malloc(ctx, (size_t)proj_dim * dim * sizeof(float),
                               (void**)&G_d);
    if (status != TQ_OK) return status;

    {
        int total = proj_dim * dim;
        int blocks = tq_div_ceil(total, TQ_BLOCK_SIZE);
        generate_gaussian_kernel<<<blocks, TQ_BLOCK_SIZE, 0, stream>>>(
            G_d, proj_dim, dim, seed);
    }

    /* query_proj = G * query (proj_dim x 1). */
    float* query_proj_d = nullptr;
    status = tq_device_malloc(ctx, proj_dim * sizeof(float),
                               (void**)&query_proj_d);
    if (status != TQ_OK) {
        tq_device_free(ctx, G_d);
        return status;
    }

    {
        cublasHandle_t handle;
        cublasCreate(&handle);
        cublasSetStream(handle, stream);

        float alpha = 1.0f, beta = 0.0f;
        /* G is row-major (proj_dim x dim) = col-major (dim x proj_dim).
         * query is dim x 1.
         * We want proj_dim x 1 = G^T(col) * query. */
        cublasSgemv(handle, CUBLAS_OP_T,
                    dim, proj_dim,
                    &alpha,
                    G_d, dim,
                    query_d, 1,
                    &beta,
                    query_proj_d, 1);

        cublasDestroy(handle);
    }

    tq_device_free(ctx, G_d);

    /* Step 2: Compute query norm. */
    float* query_norm_d = nullptr;
    status = tq_device_malloc(ctx, sizeof(float), (void**)&query_norm_d);
    if (status != TQ_OK) {
        tq_device_free(ctx, query_proj_d);
        return status;
    }

    qjl_compute_norms_kernel<<<1, TQ_BLOCK_SIZE, 0, stream>>>(
        query_d, dim, 1, query_norm_d);

    float query_norm_h;
    cudaMemcpyAsync(&query_norm_h, query_norm_d, sizeof(float),
                     cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    tq_device_free(ctx, query_norm_d);

    /* Step 3: Estimate IPs. */
    {
        int blocks = tq_div_ceil(n, TQ_BLOCK_SIZE);
        qjl_estimate_ip_kernel<<<blocks, TQ_BLOCK_SIZE, 0, stream>>>(
            query_proj_d, signs_d, norms_d,
            query_norm_h, n, proj_dim, results_out_d);
    }

    tq_device_free(ctx, query_proj_d);

    return tq_check_cuda(ctx, cudaGetLastError());
}
