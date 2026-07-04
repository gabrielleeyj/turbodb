# TurboDB Code Review: Improvements

Principal engineer review of the current implementation covering design decisions, edge cases,
correctness issues, and performance concerns across Go, CUDA, build system, and protobuf layers.

---

## Critical

### 1. Race condition in codebook cache (pkg/codebook/load.go:30-46)

The check-and-set pattern has a TOCTOU race. Between releasing the read lock and acquiring the
write lock, multiple goroutines can simultaneously load the same codebook, wasting CPU and memory.

```go
cacheMu.RLock()
if cb, ok := cache[key]; ok {
    cacheMu.RUnlock()
    return cb, nil
}
cacheMu.RUnlock()  // race window: multiple goroutines pass through here

cb, err := loadPrecomputed(dim, bitWidth)
```

**Fix:** Use double-checked locking -- re-check the cache after acquiring the write lock. Or replace
with `sync.Map` / `singleflight.Group`.

### 2. Race condition in CUDA bit-packing (cuda/src/quantize.cu:103-104)

`atomicOr` operates on a `(unsigned int*)` cast of a potentially misaligned byte address. Multiple
threads writing to adjacent bytes within the same 4-byte word produce data corruption.

```cuda
atomicOr((unsigned int*)(code_out + (byte_idx & ~3)),
         (unsigned int)(chunk << bit_off) << ((byte_idx & 3) * 8));
```

Same bug exists in `cuda/src/qjl.cu:63-64` for sign-bit extraction.

**Fix:** Use byte-level atomics or assign non-overlapping byte ranges to threads.

### 3. Race condition in FWHT butterfly (cuda/src/fwht.cu:51-56)

The shared-memory butterfly uses `if (j > i)` to deduplicate work, but thread assignment across
warp iterations can cause two threads to read/write the same shared-memory locations concurrently.

```cuda
int j = i ^ half;
if (j > i) {
    float a = smem[i];
    float b = smem[j];
    smem[i] = a + b;  // race
    smem[j] = a - b;  // race
}
```

**Fix:** Each thread should handle a deterministic, non-overlapping butterfly pair:
`for (int i = tid; i < d/2; i += blockDim.x) { int j = i + half; ... }`.

### 4. Goroutine leak in StreamQuantize (pkg/quantizer/batch.go:111-141)

If one worker encounters an error and returns, other workers remain blocked on `in <-chan` or
`out chan<-` forever if the caller doesn't close the input channel.

**Fix:** Accept `context.Context`, select on `ctx.Done()` in every worker loop, and document that
the caller MUST close the input channel.

### 5. Missing error checks after CUDA kernel launches

Throughout `fwht.cu`, `quantize.cu`, `qjl.cu`, and `search.cu`, kernel launches are not followed
by `cudaGetLastError()`. A failed launch is only caught (if at all) at the next sync point, losing
the specific kernel that failed and potentially proceeding with corrupted state.

**Fix:** Call `cudaGetLastError()` immediately after every `<<<>>>` launch.

### 6. go.mod declares Go 1.25.6 which does not exist

`go.mod:3` specifies `go 1.25.6`. The latest stable Go version is 1.23.x. This will cause build
failures on any standard toolchain.

**Fix:** Change to `go 1.22` (matching the SCOPE.md requirement).

---

## High

### 7. No `context.Context` on long-running operations

`SolveLloydMax`, `BatchQuantize`, `StreamQuantize`, and all `internal/cuda` methods lack a
`context.Context` parameter. Server code cannot cancel or time out these operations.

### 8. Unbounded batch allocation (pkg/quantizer/batch.go:16-17)

`BatchQuantize` allocates `make([]Code, len(xs))` and `make([]error, len(xs))` with no size limit.
A caller passing millions of vectors triggers unbounded memory allocation.

**Fix:** Add a max batch size constant or accept a pre-allocated output slice.

### 9. Integer overflow in PackedSize (pkg/quantizer/bitpack.go:83-85)

```go
func PackedSize(n, bitWidth int) int {
    return (bitWidth*n + 7) / 8 + 4
}
```

`bitWidth * n` overflows `int` when both are large (e.g., n=2^30, bitWidth=8).

**Fix:** Use `int64` for the intermediate computation or validate `n` against a reasonable upper bound.

### 10. Integer overflow in CUDA size calculation (cuda/src/quantize.cu:313)

```cuda
size_t total_code_bytes = (size_t)n * code_bytes;
```

The multiplication happens in `int` before the cast to `size_t`.

**Fix:** `(size_t)n * (size_t)code_bytes`.

### 11. NaN/Inf propagation not guarded

No explicit checks for NaN or Inf in vector inputs. These propagate silently through norm
computation, quantization, and inner-product estimation, producing garbage output.

**Files:** `pkg/quantizer/mse.go` (vecNorm), `pkg/quantizer/qjl.go` (EstimateIP),
`pkg/codebook/density.go` (PDF evaluation).

### 12. Missing NULL checks for optional search parameters (cuda/src/search.cu:223-229)

When `proj_dim > 0`, the search API requires `query_signs_d`, `db_signs_d`,
`query_res_norms_d`, and `db_res_norms_d` to be non-NULL. No validation exists -- the kernel
will dereference NULL.

### 13. Protobuf field validation entirely missing (api/v1/*.proto)

No field validation constraints on any message. `dimension`, `bit_width`, `top_k`, and vector
`values` length are all accepted without bounds checking. Violates the project's own security
requirements (SCOPE.md:857).

### 14. cuBLAS handle created and destroyed per call (cuda/src/qjl.cu:165-212)

Every `tq_qjl_project` call creates a new cuBLAS handle, incurring initialization overhead.
First call is especially expensive.

**Fix:** Cache the cuBLAS handle in `tq_context_t`.

### 15. LD_LIBRARY_PATH doesn't work on macOS (Makefile:93, cuda/Makefile:71)

Both Makefiles use `LD_LIBRARY_PATH` for the CUDA shared library. macOS requires
`DYLD_LIBRARY_PATH`. The build host is Darwin.

### 16. Missing go.mod dependencies

`go.mod` has zero dependencies despite SCOPE.md listing grpc, protobuf, cobra, prometheus,
otel, pgx, and others as required.

### 17. Silent K clamping in search kernel (cuda/src/search.cu:99-102)

`top_k` values above 128 are silently clamped to 128 instead of returning an error. Callers
get fewer results than requested with no indication.

---

## Medium

### 18. Panics in library code (pkg/codebook/density.go:28-30, 67-69)

`BetaDensity` and `GaussianDensity` constructors panic on invalid input instead of returning
an error. Library code should never panic on user-supplied arguments.

### 19. Zero-norm vectors silently quantize to zero (pkg/quantizer/mse.go:143-148)

A zero vector passes through `normalizeVec` and produces a zero code. When dequantized,
it always reconstructs as zero regardless of stored bits -- a silent data integrity issue.

### 20. Symmetry assumption not validated (pkg/codebook/lloyd_max.go:68-72)

Lloyd-Max solver assumes the density is symmetric and only solves the positive half. The
`Density` interface doesn't enforce this -- a non-symmetric custom density produces silently
incorrect results.

### 21. Missing input validation in QJLSketch.EstimateIP (pkg/quantizer/qjl.go:69-76)

No validation that `len(y) == q.dim` or that `signBits` length matches expectations.
Produces incorrect results or panics.

### 22. Global mutable state for codebook cache (pkg/codebook/load.go:16-19)

Package-level `var cache map[string]*Codebook` with a mutex makes testing difficult, prevents
independent cache lifetimes, and couples all callers to a single global.

**Fix:** Extract into a `CodebookCache` type that can be injected.

### 23. Excessive allocations in hot path (pkg/rotation/hadamard.go:62, 89)

Every `Apply()` call allocates a new padded buffer. For high-throughput quantization, this
generates significant GC pressure.

**Fix:** Accept a caller-provided workspace buffer or use `sync.Pool`.

### 24. Shared-memory bank conflicts in FWHT (cuda/src/fwht.cu:42-58)

The butterfly access pattern `i ^ half` causes bank conflicts when `half` is a multiple of 32.
All threads in a warp access the same shared-memory bank.

### 25. No Tensor Core utilization (cuda/src/qjl.cu)

`cublasSgemm` uses FP32. On A100/H100, using `cublasGemmEx` with
`CUBLAS_COMPUTE_32F_FAST_TF32` provides 8-20x speedup for free.

### 26. Hardcoded block size (cuda/include/turboquant_internal.h:72)

`TQ_BLOCK_SIZE 256` is fixed at compile time. Optimal block size varies by kernel and GPU
architecture. Should use `cudaOccupancyMaxPotentialBlockSize` at runtime.

### 27. Protobuf generation output path doesn't exist

`buf.gen.yaml:4` targets `../gen/go` which isn't created. `make proto` silently succeeds
without generating anything if `buf` is absent.

### 28. No protobuf breaking change enforcement

`buf.yaml` enables breaking detection but the Makefile never runs `buf breaking`. Schema
changes can silently break clients.

### 29. Overrelaxation factor is a magic number (pkg/codebook/lloyd_max.go:105)

```go
const omega = 1.7
```

No citation, reference, or explanation for why 1.7 is correct or stable.

---

## Low / Design

### 30. Duplicate exported function (pkg/quantizer/qjl.go:212)

`EstimateIP` standalone function duplicates `ProdQuantizer.EstimateInnerProduct`. One should
be removed.

### 31. Finalizers used for CUDA resource cleanup (internal/cuda/bindings.go:32, 105, 137)

`runtime.SetFinalizer` is unreliable -- runs at unpredictable times and may not run at all.
Users must call `Close()` explicitly; the finalizer should only be a safety net, and this
should be documented.

### 32. Only first batch error returned (pkg/quantizer/batch.go:115)

Error channel is buffered to `workers` but only one error is ever read. Remaining errors
from other workers are silently discarded.

### 33. String key for cache map (pkg/codebook/load.go:22)

`fmt.Sprintf("d%d_b%d", dim, bitWidth)` on every lookup. A `struct{dim, bitWidth int}` key
avoids the allocation entirely.

### 34. Binary search tie-breaking undocumented (pkg/codebook/codebook.go:66-86)

`<=` comparison means ties favor the lower centroid. This affects quantization behavior and
should be documented since it's a semantic choice.

### 35. Missing .PHONY for several Makefile targets

`test-cover`, `build-cuda`, and `test-cuda` are missing from `.PHONY`.

### 36. GoogleTest path hardcoded to /usr/local (cuda/Makefile:52)

Fails on Ubuntu (apt installs to /usr), NixOS, or non-standard Homebrew prefixes.

**Fix:** `GTEST_DIR ?= $(shell pkg-config --variable=prefix gtest 2>/dev/null || echo /usr/local)`

### 37. CUDA arch targets exclude V100/T4 (cuda/Makefile:15)

Only SM 8.0 and 9.0 are compiled. SM 7.0 (V100) and 7.5 (T4) are excluded, limiting
hardware compatibility without documentation explaining the restriction.

### 38. Dead code: compute_norm_partial (cuda/src/quantize.cu:46-53)

Device function defined but never called. Remove it.

### 39. Device context not validated in tq_destroy (cuda/src/context.cu:80-81)

`cudaSetDevice` return value is unchecked. If it fails, `cudaStreamDestroy` operates on the
wrong device or leaks the stream.

---

## Summary

| Severity | Count | Action Required |
|----------|-------|-----------------|
| Critical | 6     | Must fix before any deployment |
| High     | 11    | Fix before Phase 2 merge |
| Medium   | 12    | Fix before Phase 3 (engine) |
| Low      | 10    | Address opportunistically |
| **Total**| **39**| |

### Top 5 priorities

1. Fix all three race conditions (codebook cache, CUDA bit-packing, FWHT butterfly)
2. Add `cudaGetLastError()` after every kernel launch
3. Fix `go.mod` version and add missing dependencies
4. Add `context.Context` to all long-running Go APIs
5. Guard against NaN/Inf and validate inputs at all public API boundaries
