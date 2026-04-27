/**
 * context.cu — TurboQuant CUDA context implementation.
 *
 * Manages per-context CUDA stream, device selection, and error state.
 */

#include "turboquant.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>

/* --------------------------------------------------------------------------
 * Internal context structure
 * -------------------------------------------------------------------------- */

struct tq_context_s {
    int            device_id;
    cudaStream_t   stream;
    cublasHandle_t cublas;       /* Lazily initialized, reused across calls. */
    char           last_error[512];
};

/* --------------------------------------------------------------------------
 * Internal helpers
 * -------------------------------------------------------------------------- */

static tq_status_t check_cuda(tq_context_t ctx, cudaError_t err) {
    if (err == cudaSuccess) {
        return TQ_OK;
    }
    if (ctx != nullptr) {
        snprintf(ctx->last_error, sizeof(ctx->last_error),
                 "CUDA error %d: %s", (int)err, cudaGetErrorString(err));
    }
    if (err == cudaErrorMemoryAllocation) {
        return TQ_ERR_OOM;
    }
    return TQ_ERR_CUDA;
}

/* --------------------------------------------------------------------------
 * Context lifecycle
 * -------------------------------------------------------------------------- */

tq_status_t tq_init(int device_id, tq_context_t* out) {
    if (out == nullptr) {
        return TQ_ERR_INVALID_ARG;
    }

    tq_context_t ctx = (tq_context_t)calloc(1, sizeof(struct tq_context_s));
    if (ctx == nullptr) {
        return TQ_ERR_OOM;
    }

    ctx->device_id = device_id;
    ctx->last_error[0] = '\0';

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        tq_status_t s = check_cuda(ctx, err);
        free(ctx);
        return s;
    }

    err = cudaStreamCreate(&ctx->stream);
    if (err != cudaSuccess) {
        tq_status_t s = check_cuda(ctx, err);
        free(ctx);
        return s;
    }

    *out = ctx;
    return TQ_OK;
}

void tq_destroy(tq_context_t ctx) {
    if (ctx == nullptr) {
        return;
    }
    cudaSetDevice(ctx->device_id);
    if (ctx->cublas != nullptr) {
        cublasDestroy(ctx->cublas);
    }
    cudaStreamDestroy(ctx->stream);
    free(ctx);
}

tq_status_t tq_sync(tq_context_t ctx) {
    if (ctx == nullptr) {
        return TQ_ERR_NOT_INIT;
    }
    return check_cuda(ctx, cudaStreamSynchronize(ctx->stream));
}

const char* tq_last_error(tq_context_t ctx) {
    if (ctx == nullptr) {
        return "context is NULL";
    }
    return ctx->last_error;
}

/* --------------------------------------------------------------------------
 * Device memory helpers
 * -------------------------------------------------------------------------- */

tq_status_t tq_device_malloc(tq_context_t ctx, size_t size, void** ptr_out) {
    if (ctx == nullptr) return TQ_ERR_NOT_INIT;
    if (ptr_out == nullptr || size == 0) return TQ_ERR_INVALID_ARG;
    return check_cuda(ctx, cudaMalloc(ptr_out, size));
}

tq_status_t tq_device_free(tq_context_t ctx, void* ptr_d) {
    if (ctx == nullptr) return TQ_ERR_NOT_INIT;
    if (ptr_d == nullptr) return TQ_OK;
    return check_cuda(ctx, cudaFree(ptr_d));
}

tq_status_t tq_memcpy_h2d(tq_context_t ctx, void* dst_d,
                           const void* src_h, size_t size) {
    if (ctx == nullptr) return TQ_ERR_NOT_INIT;
    if (dst_d == nullptr || src_h == nullptr) return TQ_ERR_INVALID_ARG;
    return check_cuda(ctx, cudaMemcpyAsync(dst_d, src_h, size,
                                            cudaMemcpyHostToDevice,
                                            ctx->stream));
}

tq_status_t tq_memcpy_d2h(tq_context_t ctx, void* dst_h,
                           const void* src_d, size_t size) {
    if (ctx == nullptr) return TQ_ERR_NOT_INIT;
    if (dst_h == nullptr || src_d == nullptr) return TQ_ERR_INVALID_ARG;
    return check_cuda(ctx, cudaMemcpyAsync(dst_h, src_d, size,
                                            cudaMemcpyDeviceToHost,
                                            ctx->stream));
}

/* --------------------------------------------------------------------------
 * Device info
 * -------------------------------------------------------------------------- */

tq_status_t tq_device_memory_info(tq_context_t ctx,
                                   size_t* free_bytes,
                                   size_t* total_bytes) {
    if (ctx == nullptr) return TQ_ERR_NOT_INIT;
    if (free_bytes == nullptr || total_bytes == nullptr) return TQ_ERR_INVALID_ARG;

    cudaError_t err = cudaSetDevice(ctx->device_id);
    if (err != cudaSuccess) return check_cuda(ctx, err);

    return check_cuda(ctx, cudaMemGetInfo(free_bytes, total_bytes));
}

tq_status_t tq_device_compute_capability(tq_context_t ctx, int* cc_out) {
    if (ctx == nullptr) return TQ_ERR_NOT_INIT;
    if (cc_out == nullptr) return TQ_ERR_INVALID_ARG;

    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, ctx->device_id);
    if (err != cudaSuccess) return check_cuda(ctx, err);

    *cc_out = prop.major * 10 + prop.minor;
    return TQ_OK;
}

/* Expose stream for use by other .cu files */
cudaStream_t tq_get_stream(tq_context_t ctx) {
    return ctx ? ctx->stream : nullptr;
}

int tq_get_device_id(tq_context_t ctx) {
    return ctx ? ctx->device_id : -1;
}

/* Lazily create and cache a cuBLAS handle, bound to the context's stream. */
cublasHandle_t tq_get_cublas(tq_context_t ctx) {
    if (ctx == nullptr) return nullptr;
    if (ctx->cublas == nullptr) {
        cublasCreate(&ctx->cublas);
        cublasSetStream(ctx->cublas, ctx->stream);
    }
    return ctx->cublas;
}
