# TurboQuant Algorithm Reference

Reference: arXiv:2504.19874v1

## Core Insight

TurboQuant is a data-oblivious (zero-calibration) vector quantization scheme.
It uses random rotation followed by scalar quantization to achieve near-optimal
distortion without needing to see the data distribution.

## Algorithm 1: QuantMSE

**Goal:** Minimize MSE between original and reconstructed vectors.

1. **Rotate**: Apply a random rotation matrix `Pi` (seeded, deterministic).
   - Production path: Randomized Fast Walsh-Hadamard Transform (O(d log d)).
   - Fallback: QR decomposition of a Gaussian matrix (O(d^2)).
2. **Quantize**: For each coordinate of the rotated vector, find the nearest
   centroid in a precomputed Lloyd-Max codebook.
3. **Pack**: Store centroid indices as a packed bit-stream (b bits per coordinate).

Dequantization is the reverse: unpack -> lookup centroids -> inverse rotation.

### Codebook Generation

The codebook is generated via Lloyd-Max quantization over the Beta distribution:

```
f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
```

For high dimensions (d >= 256), this is well-approximated by N(0, 1/d).

Key property: codebooks depend only on (dimension, bit_width), not on the data.

**Implementation details (`pkg/codebook`):**

- `density.go`: `BetaDensity` (log-gamma to avoid overflow) and `GaussianDensity` (N(0, 1/d)). Auto-selected by `DensityForDim(d)` — Gaussian for d >= 256, Beta otherwise.
- `lloyd_max.go`: `SolveLloydMax` with three convergence accelerations:
  1. **Symmetry exploitation** — solves only the positive half of the symmetric distribution, mirrors for negatives.
  2. **Quantile-based initialization** — trapezoidal CDF + linear interpolation for better starting centroids.
  3. **Overrelaxation** (omega = 1.7) — steps 1.7x in the update direction, accelerating linear convergence.
  - Convergence criterion: max centroid movement < 1e-5, with 20,000 quadrature points (composite Simpson's rule).
  - Converges in < 100 iterations for all standard (d, b) pairs.
- `load.go`: Thread-safe cache (`sync.RWMutex`) → precomputed embed → on-the-fly generation fallback.
- `precomputed/`: 56 JSON codebooks for dims {128, 256, 512, 768, 1024, 1536, 3072, 4096} × bit-widths {1, 2, 3, 4, 5, 6, 8}.

## Algorithm 2: QuantProd

**Goal:** Minimize inner-product estimation error (unbiased estimator).

1. Apply QuantMSE with bit-width `b-1`.
2. Compute residual `r = x - Dequantize(Quantize(x))`.
3. Apply QJL (Quantized Johnson-Lindenstrauss) transform to residual:
   - Project through a random Gaussian matrix.
   - Store only the signs (1-bit sketch) plus residual norm.

The inner product `<x, y>` can be estimated directly from the quantized
representations without full dequantization.

## Key Theorems

**Theorem 1 (MSE bound):** The expected squared error of QuantMSE with b-bit
codebook satisfies a tight bound dependent on the Lloyd-Max distortion for the
Beta(d) distribution.

**Theorem 2 (IP variance):** The QuantProd estimator is unbiased, with variance
bounded by a term that decreases with bit-width.

## Implementation Details

### Rotation (`pkg/rotation`)

`HadamardRotator` implements the randomized FWHT:
- **Apply**: pad to next power-of-2 → sign1 multiply → in-place butterfly FWHT → normalize (1/√padDim) → sign2 multiply.
- **ApplyTranspose**: sign2 → FWHT → normalize → sign1 → truncate to original dim.
- Sign vectors are generated deterministically from a seed via PCG RNG.
- `OutDim()` returns the padded dimension (e.g., 1536 → 2048). MSE quantizer operates on `OutDim()` coordinates.
- Benchmark: ~9µs/vector at d=1536 on Apple M3 Pro. 1 allocation (padded buffer).
- Serialization: 13-byte `MarshalBinary` (type tag + dim + seed). Matrix is regenerated on load.

### MSE Quantizer (`pkg/quantizer`)

`MSEQuantizer` implements Algorithm 1:
1. Compute `‖x‖₂`, normalize to unit vector.
2. Apply rotation (output dimension may differ from input).
3. Per-coordinate binary search on sorted codebook centroids (`NearestIndex`).
4. Bit-pack indices into `ceil(b × OutDim / 8) + 4` bytes (4 bytes for stored norm as float32).

Bit packing (`bitpack.go`): LSB-first, handles arbitrary bit-widths 1–8 crossing byte boundaries.

### ProdQuantizer (`pkg/quantizer`)

`ProdQuantizer` implements Algorithm 2:
1. MSE quantize at `b-1` bits.
2. Dequantize to get `x̂`, compute residual `r = x - x̂`.
3. `QJLSketch.Sign(r)` → 1-bit sketch (sign vector) + `‖r‖₂`.
4. `EstimateInnerProduct(y, code)`: `<y, x̂>` + QJL correction term.

QJL correction: recompute the same seeded Gaussian projection for `y`, count sign agreements with stored sketch, estimate cosine similarity, scale by norms.

Empirical validation (d=256, b=4): mean estimation error ~0, variance ~1.4e-4.

### Batch API (`pkg/quantizer`)

- `BatchQuantize`: worker pool (channels + `sync.WaitGroup`), sized to `runtime.NumCPU()`.
- `BatchEstimateIP`: parallel over queries, each computing IP against all codes.
- `StreamQuantize`: channel-based streaming with graceful close propagation.
- All verified data-race-free under `go test -race`.

## GPU Implementation (`cuda/`, `internal/cuda/`)

The CUDA kernel layer reimplements all TurboQuant primitives for GPU, targeting A100 (SM 8.0) and H100 (SM 9.0). The CPU reference in `pkg/` is the correctness oracle.

### C ABI (`turboquant.h`)

All GPU operations go through a C ABI with opaque handles:
- `tq_context_t` — owns a CUDA stream (one per thread/goroutine).
- `tq_codebook_t` — device-side centroid array.
- `tq_rotator_t` — device-side sign-flip array + dim metadata.
- All functions return `tq_status_t` (never throw across the ABI boundary).

### FWHT Kernel (`fwht.cu`)

Two paths based on dimension:

1. **Small d (≤4096)**: single kernel per vector, butterfly stages in shared memory. Sign flips (S1, S2) are fused into the load/store. `log2(d)` butterfly stages with `__syncthreads()` barriers.
2. **Large d (>4096)**: multi-pass global-memory approach. Separate sign-flip kernel + `log2(d)` butterfly kernel launches.

Inverse: identical butterfly (Hadamard is self-inverse) with swapped sign order and `1/d` scaling fused into the final store.

### Quantize/Dequantize Kernels (`quantize.cu`)

- **Codebook in shared memory**: loaded once per block (≤1 KB for b≤8). Each thread processes multiple coordinates via strided loop.
- **Binary search**: per-coordinate nearest-centroid lookup, identical algorithm to CPU `NearestIndex`.
- **Bit packing**: atomic byte-level packing to handle cross-byte boundaries. Output: `ceil(b × outDim / 8)` bytes per vector.
- **Norm pipeline**: `compute_norms_kernel` → `normalize_kernel` → quantize → norms stored alongside codes.
- **Dequantize**: unpack → centroid lookup → inverse FWHT → `rescale_kernel` → `cudaMemcpy2DAsync` for strided dim truncation.

### QJL Kernel (`qjl.cu`)

- **Gaussian matrix**: generated on-device using `curand_init` with Philox4 PRNG, seeded deterministically per element `(seed, row, col)`.
- **Projection**: `cublasSgemm` (GEMM) computes `G × vectors^T` in one call. More portable and often faster than custom matmul for this shape.
- **Sign extraction**: fused post-GEMM kernel using atomicOr for packed bit output.
- **IP estimation**: per-database-vector kernel recomputes query projection signs and counts agreements against stored sketches.

### Search Kernel (`search.cu`)

- One block per query, threads strided over database vectors.
- Per-coordinate code unpacking + shared-memory codebook lookup → dot product accumulation → norm scaling.
- Thread-local min-heap (max K=128) for top-K candidates.
- Host-side merge of per-thread heaps via insertion sort (small K).

### Go Bindings (`internal/cuda/`)

- `bindings.go` (build tag `cuda`): cgo wrappers with host↔device transfers. Each method allocates device buffers, uploads, runs kernel, downloads, frees.
- `stub.go` (build tag `!cuda`): returns `ErrNoCUDA` for all operations. Allows `go build` without CUDA toolkit.
- `pool.go`: `Pool` manages multiple contexts for concurrent goroutine access.
- `runtime.SetFinalizer` on all handles for safety, but explicit `Close()` is preferred.
