# Architecture

## Package Structure (Phase 1)

```
pkg/
├── codebook/          Precomputed Lloyd-Max codebooks
│   ├── codebook.go        Codebook type, NearestIndex (binary search)
│   ├── density.go         BetaDensity, GaussianDensity (PDF + support)
│   ├── lloyd_max.go       SolveLloydMax solver (symmetry, overrelaxation)
│   ├── load.go            Thread-safe cache + embed + on-the-fly generation
│   ├── embed.go           go:embed directive for precomputed/*.json
│   └── precomputed/       56 JSON codebooks (8 dims × 7 bit-widths)
│
├── rotation/          Random rotation matrices
│   ├── rotation.go        Rotator interface + UnmarshalRotator
│   └── hadamard.go        HadamardRotator (randomized FWHT)
│
└── quantizer/         Quantization algorithms
    ├── quantizer.go       Quantizer interface, Code type
    ├── bitpack.go         PackBits / UnpackBits (arbitrary bit-widths)
    ├── mse.go             MSEQuantizer (Algorithm 1)
    ├── qjl.go             QJLSketch + ProdQuantizer (Algorithm 2)
    └── batch.go           BatchQuantize, BatchEstimateIP, StreamQuantize
```

## Data Flow

### Quantization (Algorithm 1 — MSE)

```
Input vector x (dim=d)
    │
    ├── Compute ‖x‖₂, normalize
    │
    ├── HadamardRotator.Apply(x_normalized)
    │   └── pad → sign1 → FWHT → normalize → sign2
    │   └── Output: rotated vector (dim=OutDim, may be > d)
    │
    ├── Per-coordinate: Codebook.NearestIndex (binary search)
    │   └── Output: index array (OutDim integers, each 0..2^b-1)
    │
    └── PackBits(indices, bitWidth)
        └── Output: Code { Indices []byte, Norm float32, BitWidth, Dim }
```

### Quantization (Algorithm 2 — Prod)

```
Input vector x (dim=d)
    │
    ├── MSEQuantizer.Quantize(x) at b-1 bits → mseCode
    │
    ├── MSEQuantizer.Dequantize(mseCode) → x̂
    │
    ├── residual r = x - x̂
    │
    ├── QJLSketch.Sign(r) → signBits, ‖r‖₂
    │
    └── Output: ProdCode { MSECode, SignBits, ResidualNorm, ProjDim }
```

### Inner Product Estimation

```
Query y, ProdCode for x
    │
    ├── MSEQuantizer.Dequantize(code.MSECode) → x̂
    │
    ├── <y, x̂>  (MSE component)
    │
    ├── QJLSketch.EstimateIP(code.SignBits, code.ResidualNorm, y)
    │   └── Recompute projection of y with same seed
    │   └── Count sign agreements → cosine estimate → scale by norms
    │
    └── Output: <y, x̂> + QJL correction ≈ <y, x>  (unbiased)
```

## Key Interfaces

```go
// Rotator — pkg/rotation
type Rotator interface {
    Apply(x []float32) []float32          // rotate (may change dimension)
    ApplyTranspose(x []float32) []float32 // inverse rotation
    Seed() uint64
    Dim() int                             // input dimension
    OutDim() int                          // output dimension (>= Dim)
    MarshalBinary() ([]byte, error)
}

// Quantizer — pkg/quantizer
type Quantizer interface {
    Quantize(x []float32) (Code, error)
    Dequantize(c Code) ([]float32, error)
    BitWidth() int
    Dim() int
    Codebook() *codebook.Codebook
    Rotator() rotation.Rotator
}
```

## Immutability

All core types are immutable after construction:
- `Codebook`: defensive copy of centroids in constructor and getter.
- `HadamardRotator`: sign vectors and config are fixed post-`New`.
- `MSEQuantizer`, `ProdQuantizer`: hold references to immutable sub-components.

This guarantees goroutine safety without locks for `Apply`, `Quantize`, and `EstimateInnerProduct`.

## Concurrency Model

- **Single-vector operations** are lock-free (immutable state).
- **Batch operations** use a channel-based worker pool sized to `runtime.NumCPU()`.
- **Codebook cache** uses `sync.RWMutex` (read-heavy, write-rare).

## Phase 2: CUDA Kernel Layer

```
cuda/
├── include/
│   ├── turboquant.h           C ABI header (opaque handles, tq_status_t)
│   └── turboquant_internal.h  Shared internals between .cu files
├── src/
│   ├── context.cu             Context lifecycle, device memory helpers
│   ├── codebook.cu            Device-side codebook + rotator management
│   ├── fwht.cu                Batched Fast Walsh-Hadamard Transform
│   ├── quantize.cu            MSE quantize/dequantize kernels
│   ├── qjl.cu                 QJL sketch via cuBLAS GEMM
│   └── search.cu              Brute-force top-K search
├── tests/
│   ├── test_fwht.cu           FWHT round-trip, norm, determinism
│   └── test_quantize.cu       Quantize/dequantize parity
└── Makefile                   Builds libturboquant_cuda.so

internal/cuda/
├── context.go       Context/Codebook/Rotator interfaces + SearchParams
├── errors.go        tq_status_t → Go error mapping
├── types.go         SearchResult, DeviceInfo
├── bindings.go      cgo wrappers (build tag: cuda)
├── stub.go          CPU fallback (build tag: !cuda)
├── pool.go          Context pool for concurrent use
└── cuda_test.go     Stub + error mapping tests
```

### CUDA Design Decisions

- **One context per stream**: `tq_context_t` owns a CUDA stream. Not thread-safe; use the `Pool` for concurrency.
- **Build tag isolation**: `go build` (no tag) compiles cleanly without CUDA. `go build -tags cuda` links to `libturboquant_cuda.so`.
- **Shared-memory codebook**: codebook fits in shared memory for b≤8 (max 256 floats = 1 KB), enabling fast per-coordinate binary search.
- **Two FWHT paths**: small d (≤4096) uses shared-memory butterfly; large d uses global-memory multi-pass.
- **QJL via cuBLAS**: seeded Gaussian matrix generated on GPU (cuRAND Philox), projected via `cublasSgemm`, signs extracted in fused kernel.
- **Top-K search**: per-thread min-heap (K≤128) in GPU kernel, host-side merge for final results.

### Data Flow (GPU Path)

```
Host vectors (n × dim)
    │
    ├── tq_memcpy_h2d → device
    ├── compute_norms_kernel → per-vector norms
    ├── pad_vectors_kernel → n × out_dim (zero-padded)
    ├── normalize_kernel → unit vectors
    ├── tq_fwht_batch → randomized Hadamard rotation
    ├── quantize_mse_kernel → packed bit-stream (shared-mem codebook)
    │
    └── tq_memcpy_d2h → host codes + norms
```

## Future Phases

```
Phase 3: cmd/turbodb-engine/      Standalone gRPC engine
Phase 4: pkg/formats/, python/    SafeTensors/GGUF + KV cache plugin
Phase 5: postgres/                PostgreSQL extension (pg_turboquant)
Phase 6: cmd/turbodb-sync/ctl/    Replication, CDC, control plane
```
