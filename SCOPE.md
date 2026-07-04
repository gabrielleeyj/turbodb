# TurboDB: Production-Scale TurboQuant Database — Full Engineering Scope

> **Project codename:** TurboDB
> **Primary language:** Go (1.22+)
> **Hardware target:** NVIDIA H100 / A100 (CUDA 12.x), x86_64 Linux
> **Architecture:** Hybrid — PostgreSQL extension + standalone GPU engine + LLM framework plugins
> **Reference algorithm:** TurboQuant (arXiv:2504.19874v1, ICLR 2026)
> **Last updated:** April 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Why Go (and where Go can't reach)](#2-why-go-and-where-go-cant-reach)
3. [System Architecture](#3-system-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Repository Layout](#5-repository-layout)
6. [Component 1: Core TurboQuant Library (`turboquant-core`)](#6-component-1-core-turboquant-library-turboquant-core)
7. [Component 2: CUDA Kernel Layer (`turboquant-cuda`)](#7-component-2-cuda-kernel-layer-turboquant-cuda)
8. [Component 3: Standalone GPU Vector Engine (`turbodb-engine`)](#8-component-3-standalone-gpu-vector-engine-turbodb-engine)
9. [Component 4: PostgreSQL Extension (`pg_turboquant`)](#9-component-4-postgresql-extension-pg_turboquant)
10. [Component 5: Format Support — SafeTensors & GGUF (`turbodb-formats`)](#10-component-5-format-support--safetensors--gguf-turbodb-formats)
11. [Component 6: KV Cache Integration (`turboquant-kv`)](#11-component-6-kv-cache-integration-turboquant-kv)
12. [Component 7: CDC & Replication (`turbodb-sync`)](#12-component-7-cdc--replication-turbodb-sync)
13. [Component 8: Control Plane & Operations (`turbodb-ctl`)](#13-component-8-control-plane--operations-turbodb-ctl)
14. [Testing Strategy](#14-testing-strategy)
15. [Security Model](#15-security-model)
16. [Deployment](#16-deployment)
17. [Phased Delivery & Milestones](#17-phased-delivery--milestones)
18. [Risks & Mitigations](#18-risks--mitigations)
19. [Success Metrics](#19-success-metrics)

---

## 1. Executive Summary

TurboDB is a GPU-accelerated database system built around the TurboQuant vector quantization algorithm. It delivers two production-grade capabilities from a unified codebase:

- **Vector similarity search** — nearest-neighbor retrieval over quantized vector indexes, with 8× compression at 4-bit and near-optimal recall, exposed both as a standalone gRPC service and as a PostgreSQL access method.
- **KV cache quantization** — drop-in compression backend for vLLM and SGLang, reducing KV cache memory by 5–6× with zero-calibration, data-oblivious quantization.

The system is written primarily in **Go**, with CUDA C++ kernels invoked via `cgo` and a thin C shim for PostgreSQL's extension API. Go handles the control plane (services, orchestration, IPC, CDC pipelines, APIs, lifecycle management) while CUDA handles the hot path (FWHT rotation, scalar quantization, QJL sketch, ANN search).

### Deliverables at a glance

| Deliverable | Format | Primary Users |
|---|---|---|
| `libturboquant.so` + Go API | Shared library + Go module | Go apps, FFI consumers |
| `turbodb-engine` binary | Linux x86_64 | Standalone deployments |
| `pg_turboquant.so` PostgreSQL extension | `.so` loaded by postgres | pgvector users |
| `turboquant-kv` Python package | PyPI wheel | vLLM/SGLang operators |
| `turbodb-ctl` CLI | Linux/macOS binary | Operators |
| SafeTensors / GGUF I/O tools | Go binaries + Python bindings | Data scientists |

---

## 2. Why Go (and where Go can't reach)

### Go's fit for this project

- **Concurrency model.** GPU daemon orchestration, IPC with N PostgreSQL backends, gRPC serving, and CDC consumption are all naturally modeled as goroutines.
- **Single-binary deployment.** The standalone engine ships as one static binary with embedded assets — no Python runtime, no JVM, no dependency hell in production.
- **gRPC + Protobuf first-class.** `google.golang.org/grpc` and `buf` give us a type-safe, cross-language API boundary for free.
- **Observability ecosystem.** Prometheus, OpenTelemetry, pprof, and structured logging (`slog`) are mature and zero-cost to adopt.
- **Memory safety for the control plane.** The bulk of the code — index metadata, segment management, WAL, replication — never touches raw GPU memory and benefits from Go's GC and race detector.

### Where Go cannot reach directly

These boundaries require non-Go code; we minimize the surface area and hide it behind Go interfaces.

| Boundary | Language | Mechanism |
|---|---|---|
| CUDA kernels (FWHT, quantize, QJL, GEMM launch) | CUDA C++ (`.cu`) | `cgo` → `libturboquant_cuda.so` |
| PostgreSQL `IndexAmRoutine` | C | Static C shim that IPCs to Go daemon via Unix socket |
| Python integration (vLLM/SGLang) | Python wrapper | CFFI over `libturboquant.so` OR gRPC to engine |
| Triton kernels (for JIT-compiled KV cache ops) | Python/Triton | Called only from Python plugin side |

**Design rule:** every non-Go component is wrapped in a Go interface at its narrowest point. Tests at the interface can run without a GPU.

---

## 3. System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          APPLICATION LAYER                                │
│  (user apps — RAG pipelines, recommendation systems, LLM serving)         │
└──────────────────────────────────────────────────────────────────────────┘
             │                      │                         │
             ▼                      ▼                         ▼
    ┌──────────────────┐  ┌──────────────────┐    ┌────────────────────┐
    │   PostgreSQL     │  │   turbodb-engine │    │  vLLM / SGLang     │
    │  + pg_turboquant │  │  (standalone)    │    │  + turboquant-kv   │
    │                  │  │                  │    │    plugin          │
    │  SQL / pgvector  │  │  gRPC API        │    │  Python runtime    │
    └──────────────────┘  └──────────────────┘    └────────────────────┘
             │                      │                         │
             │  Unix socket         │  cgo                    │ CFFI / gRPC
             ▼                      ▼                         ▼
    ┌────────────────────────────────────────────────────────────────────┐
    │                    GO CONTROL PLANE (turboquant-core)              │
    │                                                                    │
    │  • Index metadata, segment lifecycle, WAL                          │
    │  • Rotation matrix + codebook management                           │
    │  • Query planner & result merging                                  │
    │  • Replication, CDC consumption                                    │
    └────────────────────────────────────────────────────────────────────┘
                                  │  cgo
                                  ▼
    ┌────────────────────────────────────────────────────────────────────┐
    │              CUDA KERNEL LAYER (libturboquant_cuda.so)             │
    │                                                                    │
    │  • FWHT rotation (HadaCore-style)                                  │
    │  • Scalar quantize / dequantize (codebook lookup)                  │
    │  • QJL transform (sign + matmul)                                   │
    │  • Inner-product search over quantized vectors                     │
    │  • cuVS/RAFT bindings for CAGRA graph index                        │
    └────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
    ┌────────────────────────────────────────────────────────────────────┐
    │                       NVIDIA GPU (H100 / A100)                     │
    └────────────────────────────────────────────────────────────────────┘

Supporting infrastructure (all Go):
  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
  │ turbodb-sync  │  │  turbodb-ctl  │  │   Prometheus  │
  │ (CDC)         │  │  (CLI)        │  │  + Grafana    │
  └───────────────┘  └───────────────┘  └───────────────┘
```

### Deployment shapes supported

1. **Embedded in PostgreSQL only.** `pg_turboquant` extension, no standalone engine. Good for small-to-medium workloads, strict ACID, existing Postgres infra.
2. **Standalone engine only.** `turbodb-engine` serving gRPC. Good for LLM-adjacent workloads, high QPS, billion-scale indexes.
3. **Hybrid (recommended for production).** PostgreSQL as source of truth for raw vectors + metadata; `turbodb-engine` serves the quantized index; `turbodb-sync` keeps them consistent via logical replication.
4. **KV cache only.** `turboquant-kv` plugin in vLLM/SGLang, no Postgres, no engine. Lightest deployment.

---

## 4. Technology Stack

### Languages & runtimes

- **Go 1.22+** — primary language. Use generics where they clarify (e.g., quantizer typed over bit-width).
- **CUDA C++ 12.x** — kernel layer. Target SM 8.0 (A100) and SM 9.0 (H100).
- **C99** — PostgreSQL extension shim only.
- **Python 3.10+** — vLLM/SGLang plugin wrapper.

### Core Go dependencies

```
google.golang.org/grpc                  // gRPC server/client
google.golang.org/protobuf              // Protobuf runtime
github.com/spf13/cobra                  // CLI framework
github.com/prometheus/client_golang     // Metrics
go.opentelemetry.io/otel                // Tracing
github.com/stretchr/testify             // Assertions (tests only)
github.com/jackc/pgx/v5                 // Postgres client (CDC & tooling)
github.com/twmb/franz-go                // Kafka (CDC transport)
github.com/dgraph-io/badger/v4          // Embedded KV store (WAL metadata)
github.com/klauspost/compress           // zstd for segment compression
github.com/dominikbraun/graph           // Internal graph algorithms
```

Avoid: ORMs, heavy DI frameworks, reflection-heavy config libs. Prefer `stdlib` + small libraries.

### CUDA dependencies

- **cuBLAS** — dense GEMM for QR-based rotation fallback.
- **cuVS / RAFT** — CAGRA graph index primitives.
- **HadaCore reference** — for Tensor-Core FWHT (reimplemented or linked).
- **Thrust** — for sort/reduce in result merging.

### Build & tooling

- **Bazel** or **Make + `go build`** — we default to Make; Bazel is proposed only if the CUDA + Go + C shim build becomes unwieldy.
- **buf** — Protobuf linting, breaking-change detection, code generation.
- **golangci-lint** — with `gosec`, `errcheck`, `staticcheck`, `revive` enabled.
- **goimports** + `gofumpt` — formatting.

---

## 5. Repository Layout

Monorepo; one Go module at root with sub-packages. Non-Go artifacts are siblings under `cuda/` and `postgres/`.

```
turbodb/
├── go.mod
├── go.sum
├── Makefile
├── README.md
├── SCOPE.md                          # this document
├── docs/
│   ├── architecture.md
│   ├── algorithms.md                 # TurboQuant math recap
│   ├── operations.md
│   └── api/                          # generated protobuf docs
├── api/                              # Protobuf definitions
│   └── v1/
│       ├── engine.proto              # gRPC service for turbodb-engine
│       ├── index.proto               # index types, quantized vector wire format
│       ├── kv_cache.proto            # KV cache quantization service
│       └── admin.proto               # admin & control-plane
├── pkg/                              # Go code (public API)
│   ├── quantizer/                    # pure Go CPU reference impl
│   ├── codebook/                     # Lloyd-Max codebook generation
│   ├── rotation/                     # rotation matrix management (seeded)
│   ├── index/                        # index interfaces, segment management
│   ├── search/                       # ANN search primitives
│   ├── wal/                          # write-ahead log
│   ├── replication/                  # replica stream
│   ├── formats/
│   │   ├── safetensors/              # reader + writer
│   │   └── gguf/                     # reader + writer
│   └── telemetry/                    # tracing, metrics, logging
├── internal/                         # private packages
│   ├── cuda/                         # cgo bindings to CUDA layer
│   ├── pgproto/                      # PostgreSQL wire protocol for shim IPC
│   └── testutil/
├── cmd/
│   ├── turbodb-engine/               # standalone gRPC server
│   ├── turbodb-ctl/                  # CLI
│   ├── turbodb-sync/                 # CDC consumer
│   └── turbodb-bench/                # benchmarking harness
├── cuda/                             # CUDA C++ sources
│   ├── include/
│   │   └── turboquant.h              # C ABI header consumed by cgo
│   ├── src/
│   │   ├── fwht.cu                   # Fast Walsh–Hadamard Transform
│   │   ├── rotation.cu               # QR-based rotation (fallback)
│   │   ├── quantize.cu               # scalar quantization kernels
│   │   ├── qjl.cu                    # QJL sign + projection
│   │   ├── search.cu                 # inner-product scan kernels
│   │   └── cagra_bridge.cu           # cuVS/RAFT glue
│   ├── tests/                        # CUDA unit tests (GoogleTest)
│   └── Makefile
├── postgres/                         # C extension sources
│   ├── pg_turboquant.c               # AM registration & IPC shim
│   ├── pg_turboquant--0.1.sql        # extension DDL
│   ├── pg_turboquant.control
│   └── Makefile
├── python/                           # Python bindings for KV cache plugin
│   ├── turboquant_kv/
│   │   ├── __init__.py
│   │   ├── vllm_plugin.py
│   │   ├── sglang_plugin.py
│   │   └── _ffi.py                   # CFFI bindings to libturboquant.so
│   ├── pyproject.toml
│   └── tests/
├── deploy/
│   ├── docker/
│   ├── helm/
│   └── systemd/
└── test/
    ├── integration/                  # cross-component tests
    ├── e2e/                          # end-to-end with real GPU
    └── data/                         # small fixture datasets
```

---

## 6. Component 1: Core TurboQuant Library (`turboquant-core`)

**Purpose:** A pure-Go, CPU-only reference implementation of TurboQuant. This is the *source of truth* for algorithmic correctness; the CUDA layer is validated against it.

**Location:** `pkg/quantizer`, `pkg/codebook`, `pkg/rotation`

### Task 1.1 — Codebook generation (Lloyd-Max over Beta distribution)

- [x] Implement the Beta density `f_X(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1-x²)^((d-3)/2)` in `pkg/codebook/density.go` using `math/Lgamma`.
- [x] Implement the high-dimensional Gaussian approximation `N(0, 1/d)` for `d ≥ 256` in `pkg/codebook/density.go`.
- [x] Implement the iterative Lloyd-Max solver in `pkg/codebook/lloyd_max.go`:
  - Input: target distribution `f`, bit-width `b`, tolerance `ε`, max iterations.
  - Output: sorted centroids `c[0..2^b-1]`.
  - Algorithm: alternating step of (a) recompute cluster boundaries as midpoints, (b) recompute centroids as conditional means `∫ x·f(x) dx / ∫ f(x) dx` over each bucket.
  - Optimizations: symmetry exploitation (solve positive half only), overrelaxation (ω=1.7), quantile-based initialization.
- [x] Precompute codebooks for `(d, b)` pairs where `d ∈ {128, 256, 512, 768, 1024, 1536, 3072, 4096}` and `b ∈ {1, 2, 3, 4, 5, 6, 8}` (56 files).
- [x] Embed precomputed codebooks as `go:embed` JSON assets in `pkg/codebook/precomputed/`.
- [x] Public API: `codebook.Load(d, b int) (*Codebook, error)` with in-memory caching.
- [x] Validate each precomputed codebook matches TurboQuant paper Table 1 values within 1%.

**Acceptance criteria:**
- `TestLloydMaxConvergence`: distortion decreases monotonically, converges in < 100 iterations for all `(d, b)` pairs.
- `TestCodebookValues_b1`: for `b=1` and `d=1536`, centroids are within 1% of `±√(2/πd)` from the paper.
- Unit tests run in < 5s on CI without any GPU.

### Task 1.2 — Rotation matrix management

- [x] Define `pkg/rotation.Rotator` interface (with `OutDim()` for padded dimensions).
- [ ] Implement `pkg/rotation.QRRotator` — deferred; HadamardRotator is sufficient for Phase 1. Will add with `gonum` dependency when needed.
- [x] Implement `pkg/rotation.HadamardRotator` — randomized FWHT (sign flip → Hadamard → sign flip). Pads `d` up to next power of 2. ~9 µs/vector on Apple M3 Pro.
- [x] Persistence: `Rotator.MarshalBinary` / `UnmarshalRotator` — serialize seed, dim, type tag. Full matrix is never stored; it's regenerated on load.
- [x] Thread-safety: `Apply` is goroutine-safe without locks. Rotators are immutable post-construction.

**Acceptance criteria:**
- `TestRotationPreservesNorm`: ‖Πx‖ = ‖x‖ within 1e-5 for 1000 random unit vectors.
- `TestRotationRoundTrip`: `Π^T · Π · x = x` within 1e-5.
- `TestRotationDeterminism`: same seed + same input produces identical output across process restarts.
- `BenchmarkHadamardRotator_d1536`: < 15 µs per vector on a modern x86 core (CPU baseline).

### Task 1.3 — MSE quantizer (`QuantMSE`)

- [x] Define `pkg/quantizer.Quantizer` interface and `Code` type.
- [x] Implement `pkg/quantizer.MSEQuantizer` following Algorithm 1 of the paper:
  - `Quantize`: rotate → for each coordinate, binary search on sorted codebook for nearest centroid → pack indices into bit-stream.
  - `Dequantize`: unpack indices → lookup centroids → inverse rotation.
- [x] Implement tight bit packing in `pkg/quantizer/bitpack.go`: a 4-bit, d=2048 (padded from 1536) vector is exactly 1024 bytes; a 3-bit vector is 768 bytes. Manual shift-and-mask with round-trip tests.
- [x] Handle non-unit-norm vectors: pre-compute and store norm separately, normalize before quantization, rescale after dequantization. Store norm as `float32`.

**Acceptance criteria:**
- `TestMSEQuantizerRoundTrip`: for random unit vectors, `‖x - Dequantize(Quantize(x))‖²` matches Theorem 1 bounds within 10% for `b ∈ {1,2,3,4,5}`.
- `TestBitPackingExact`: quantized size is exactly `ceil(b*d/8) + 4` bytes (4 bytes for stored norm).
- Fuzzing with `testing/fuzz`: no panics for arbitrary-length inputs, proper errors for dimension mismatches.

### Task 1.4 — Inner-product quantizer (`QuantProd`)

- [x] Implement `pkg/quantizer.QJLSketch` — the 1-bit QJL transform from Definition 1 of the paper. Seeded Gaussian projection via `math/rand/v2` PCG, outputs sign vector + norm. Deterministic (`TestQJLSketchDeterminism`).
- [x] Implement `pkg/quantizer.ProdQuantizer` — composition per Algorithm 2:
  1. Apply `MSEQuantizer` with bit-width `b-1`.
  2. Compute residual `r = x - Dequantize(Quantize(x))`.
  3. Apply `QJLSketch` to `r`, store sign vector + `‖r‖₂`.
- [x] Implement `EstimateInnerProduct(y, code) float32` that reconstructs the unbiased inner-product estimate directly from the quantized representation, without full dequantization (faster hot path). Verified unbiased (mean error ~0, variance 1.4e-4 at d=256 b=4).

**Acceptance criteria:**
- `TestProdQuantizerUnbiased`: mean of `EstimateInnerProduct(y, Quantize(x)) - <y, x>` over 10k samples is statistically indistinguishable from zero (Welch's t-test, p > 0.05).
- `TestProdQuantizerVariance`: empirical variance matches Theorem 2 within 20%.

### Task 1.5 — Batch API

- [x] `BatchQuantize(xs [][]float32) ([]Code, error)` — worker pool sized to `runtime.NumCPU()` via channels + `sync.WaitGroup`. Verified deterministic match with individual quantization.
- [x] `BatchEstimateIP(queries [][]float32, codes []ProdCode) [][]float32` — returns an `N_q × N_c` matrix. Parallel over queries. Verified match with single estimation within 1e-6.
- [x] Streaming variant: `StreamQuantize(in <-chan []float32, out chan<- Code)` for bulk ingest. Worker pool with graceful channel close.

**Acceptance criteria:**
- `BenchmarkBatchQuantize_1M_d1536_b4`: throughput ≥ 500 vectors/sec/core on x86 (CPU baseline target; GPU will be 1000× faster).
- No data races under `go test -race`.

---

## 7. Component 2: CUDA Kernel Layer (`turboquant-cuda`)

**Purpose:** High-performance GPU implementations of TurboQuant primitives. Built as a shared library `libturboquant_cuda.so` exposing a C ABI consumed by Go via `cgo`.

**Location:** `cuda/`, Go bindings in `internal/cuda/`

### Task 2.1 — C ABI header design

- [x] Define `cuda/include/turboquant.h` with opaque handles and explicit error codes:
  ```c
  typedef struct tq_context_s* tq_context_t;
  typedef struct tq_codebook_s* tq_codebook_t;
  typedef struct tq_rotator_s*  tq_rotator_t;

  typedef enum {
      TQ_OK = 0,
      TQ_ERR_CUDA = 1,
      TQ_ERR_OOM = 2,
      TQ_ERR_INVALID_ARG = 3,
      TQ_ERR_DIM_NOT_POW2 = 4,
      // ...
  } tq_status_t;

  tq_status_t tq_init(int device_id, tq_context_t* out);
  void        tq_destroy(tq_context_t ctx);

  tq_status_t tq_quantize_mse_batch(
      tq_context_t ctx,
      const float* vectors_d, int n, int d,
      tq_rotator_t rot, tq_codebook_t cb,
      uint8_t* codes_out_d, float* norms_out_d);
  // ... etc
  ```
- [x] All pointers ending in `_d` are device pointers; callers manage allocation.
- [x] Return `tq_status_t` everywhere; never throw across the ABI boundary.
- [x] Thread safety: one `tq_context_t` per CUDA stream; document this contract.

### Task 2.2 — Fast Walsh–Hadamard Transform kernel

- [x] Implement `cuda/src/fwht.cu` with shared-memory butterfly FWHT (Tensor-Core variant deferred to perf tuning).
- [x] Support batched FWHT: `N × d` input matrix, where `d` is a power of 2 up to 65,536.
- [x] Signed variant: apply `±1` diagonal sign matrix before and after to produce the randomized Hadamard rotation.
- [x] Fuse the sign multiplication with the first/last butterfly stage (small-d path fuses in shared-memory kernel).
- [ ] Validate numerical parity with CPU reference to 1e-4 relative error. (requires GPU CI)

**Performance target:** ≥ 1.1 TB/s effective throughput on H100 for `d=4096, N=1M`.

### Task 2.3 — Quantize / Dequantize kernels

- [x] Implement `cuda/src/quantize.cu`:
  - `quantize_mse_kernel` — one thread per coordinate, binary-search codebook loaded to shared memory, write packed bits. Codebook fits entirely in shared memory for `b ≤ 8` (256 floats = 1 KB).
  - `dequantize_mse_kernel` — inverse: unpack bits, lookup centroid, write float.
- [x] Implement bit packing/unpacking for all bit-widths (1-8). Uses atomic byte-level packing.
- [x] Fuse with rotation: `tq_quantize_rotate_batch` delegates to two-step (fused single-kernel deferred to perf tuning).

**Performance target:** ≥ 50M vectors/sec at `d=1536, b=4` on H100.

### Task 2.4 — QJL kernel

- [x] Implement `cuda/src/qjl.cu`:
  - GEMM against seeded Gaussian matrix (use cuBLAS for portability).
  - Sign extraction as a single fused post-op.
  - Optionally substitute GEMM with another FWHT for the structured-Gaussian variant (deferred to perf tuning).

### Task 2.5 — Search kernels

- [x] Implement `cuda/src/search.cu`:
  - `tq_search_brute_force` — given `N_q` query codes and `N_db` database codes, compute all inner-product estimates. Uses shared-memory codebook + per-thread min-heap for top-K.
  - Host-side merge for final top-K reduction.
- [ ] For scale, integrate cuVS CAGRA graph index (deferred to Phase 3 engine work).

**Performance target:** ≥ 200k QPS for top-10 over 1M vectors at `d=1536, b=4` on H100.

### Task 2.6 — Go `cgo` bindings

- [x] `internal/cuda/bindings.go` — one-to-one wrappers around the C ABI.
- [x] `internal/cuda/context.go` — Go-friendly `Context` interface with `Close()` for deterministic cleanup.
- [x] `internal/cuda/errors.go` — map `tq_status_t` to Go error types.
- [x] Build tag `cuda` — non-GPU build compiles without CUDA runtime (falls back to CPU reference via `stub.go`).
- [x] `internal/cuda/pool.go` — Go-side context pool to avoid fragmentation and bound GPU memory per-process.

**Acceptance criteria:**
- [ ] `TestCUDAParity`: for 10k random unit vectors, CUDA output matches CPU reference to `1e-4` max absolute error. (requires GPU CI)
- [ ] `TestCUDANoLeaks`: after 1M quantize/dequantize cycles, GPU memory usage is stable. (requires GPU CI)
- [x] Builds cleanly with `go build -tags cuda` and `go build` (no tag). ✅ Verified (no-tag build passes on macOS).

---

## 8. Component 3: Standalone GPU Vector Engine (`turbodb-engine`)

**Purpose:** A gRPC-accessible vector database with GPU-accelerated TurboQuant indexes. Billion-scale, ACID-light (atomic commits per operation, eventual consistency across replicas).

**Location:** `cmd/turbodb-engine/`, `pkg/index/`, `pkg/search/`, `pkg/wal/`

### Task 3.1 — Segment architecture

Adopt Milvus's growing/sealed segment model — it proved necessary for high-churn workloads.

- [ ] Define `pkg/index.Segment` interface with concrete types:
  - `GrowingSegment` — mutable, append-only, stores raw vectors + metadata. Queries use brute-force.
  - `SealedSegment` — immutable, stores quantized index only. Queries use TurboQuant search kernels.
- [ ] `pkg/index.Collection` — manages segments for one logical index:
  - Active `GrowingSegment` accepts writes.
  - Background sealer thread rotates segments once they reach a size threshold (default 1M vectors) and builds the quantized index on GPU.
  - Tombstone log for deletes — applied at compaction time.
- [ ] Segment file format (custom, not SafeTensors for this internal use):
  - Fixed header: magic, version, dim, bit-width, rotator seed, codebook ID, vector count.
  - Body: packed quantized codes, residual QJL sketches (for `ProdQuantizer`), per-vector norms, vector IDs.
  - Footer: CRC32C, trailer length.
- [ ] Memory-map sealed segments for zero-copy loading on startup.

### Task 3.2 — Write-Ahead Log

- [ ] `pkg/wal/` — simple append-only log on disk, one file per 100MB, with fsync policy (configurable: every write vs. group commit every 10ms).
- [ ] Records: `OpInsert(id, vector)`, `OpDelete(id)`, `OpSegmentSealed(seg_id, file_path)`, `OpCheckpoint(lsn)`.
- [ ] Recovery on startup: replay from last checkpoint LSN.
- [ ] WAL truncation: once a checkpoint is durable, old WAL files are deleted.

### Task 3.3 — gRPC API

- [ ] Define service in `api/v1/engine.proto`:
  ```proto
  service TurboDBEngine {
    rpc CreateCollection(CreateCollectionRequest) returns (CreateCollectionResponse);
    rpc DropCollection(DropCollectionRequest) returns (DropCollectionResponse);
    rpc Insert(InsertRequest) returns (InsertResponse);
    rpc InsertBatch(stream InsertBatchRequest) returns (InsertBatchResponse);
    rpc Delete(DeleteRequest) returns (DeleteResponse);
    rpc Search(SearchRequest) returns (SearchResponse);
    rpc SearchBatch(SearchBatchRequest) returns (stream SearchBatchResponse);
    rpc GetStats(GetStatsRequest) returns (GetStatsResponse);
    rpc Flush(FlushRequest) returns (FlushResponse);
  }
  ```
- [ ] Unary and streaming variants for bulk operations.
- [ ] Authentication via mTLS; authorization via a simple RBAC table persisted in Badger.
- [ ] Deadline propagation using context.Context throughout.

### Task 3.4 — Query planner

- [ ] For each `Search` request, the planner:
  1. Estimates candidates per segment based on size and historical recall.
  2. Issues parallel search goroutines, one per sealed segment (each uses a pinned CUDA stream from the pool).
  3. Brute-forces the growing segment on CPU or GPU depending on its size.
  4. Merges top-K results using a bounded min-heap.
- [ ] Implement adaptive K-oversearch: request `K · oversearch_factor` candidates from each segment, then rerank the combined set against full-precision vectors (if the user requested `rerank=true`).
- [ ] Expose query hints: `ef_search`-equivalent (for CAGRA-based segments), `exact=true` for brute-force fallback.

### Task 3.5 — Memory management

- [ ] GPU memory budget enforced per collection via a semaphore (`golang.org/x/sync/semaphore`).
- [ ] Segment pinning: hot segments stay in GPU memory; cold ones spill to host pinned memory, then to NVMe.
- [ ] Prefetch policy: on multi-query workloads, speculatively pin segments predicted to be needed.

### Task 3.6 — Observability

- [ ] Prometheus metrics exposed at `/metrics`:
  - `turbodb_search_latency_seconds` (histogram, with quantile labels)
  - `turbodb_insert_throughput_vectors_total` (counter)
  - `turbodb_segments_sealed_total`, `turbodb_segments_active`
  - `turbodb_gpu_memory_bytes`, `turbodb_host_memory_bytes`
  - `turbodb_wal_fsync_latency_seconds`
- [ ] OpenTelemetry tracing: spans for `Search` → planner → per-segment search → merge.
- [ ] Structured logs via `log/slog`; JSON output in production, text in dev.

**Acceptance criteria:**
- `TestEngineRecovery`: crash-and-recover preserves all committed writes, drops uncommitted in-flight writes.
- `TestEngineConcurrentReadWrite`: 10k concurrent readers + 1k writers for 60s with no data races, no lost writes, eventually consistent results.
- `BenchmarkSearchEndToEnd_1M_d1536_top10`: p50 latency < 5ms, p99 < 20ms on single H100.

---

## 9. Component 4: PostgreSQL Extension (`pg_turboquant`)

**Purpose:** Register a custom PostgreSQL access method that uses TurboQuant for compression. Delegates GPU operations to `turbodb-engine` via Unix-socket IPC (PG-Strom pattern).

**Location:** `postgres/` (C), IPC server in Go at `cmd/turbodb-engine/` (extend existing)

### Task 4.1 — Extension skeleton

- [ ] `postgres/pg_turboquant.control` — extension metadata.
- [ ] `postgres/pg_turboquant--0.1.sql` — DDL registering opclasses, operators, access method.
- [ ] `postgres/Makefile` — uses PGXS for portable builds.
- [ ] `postgres/pg_turboquant.c` — declare `PG_MODULE_MAGIC`, `_PG_init()`.
- [ ] Depend on `vector` type from pgvector for data compatibility — our index wraps their type, doesn't replace it.

### Task 4.2 — `IndexAmRoutine` implementation

Register a `turboquant` access method with the following function pointers (all defined in C, all IPC to Go daemon):

- [ ] `ambuild` — initial index build. Sends raw vectors over Unix socket to engine, engine returns a handle stored in PostgreSQL metapage.
- [ ] `ambuildempty` — create empty index for unlogged tables.
- [ ] `aminsert` — per-tuple insertion. Sends vector over IPC; engine quantizes and stages.
- [ ] `ambulkdelete` — bulk delete given a VACUUM tid set. Passes tids to engine.
- [ ] `amvacuumcleanup` — trigger engine-side compaction.
- [ ] `ambeginscan` / `amgettuple` / `amendscan` — query-time: send query vector over IPC, receive back a stream of tids ordered by similarity.
- [ ] `amoptions` — index options: `bits = 4`, `oversearch_factor = 2`, `use_qjl = true`.
- [ ] `amvalidate`, `amcostestimate` — minimum viable implementations.

### Task 4.3 — IPC protocol

- [ ] Define a binary framed protocol in `internal/pgproto/`:
  - 4-byte length prefix.
  - 2-byte opcode.
  - Payload — little-endian, schema-versioned.
- [ ] Opcodes: `BUILD_BEGIN`, `BUILD_VECTOR`, `BUILD_COMMIT`, `INSERT`, `DELETE`, `SEARCH_BEGIN`, `SEARCH_NEXT`, `SEARCH_END`, `STATS`, `SHUTDOWN`.
- [ ] Transport: `SOCK_STREAM` Unix socket at `/var/run/turbodb/engine.sock` (configurable).
- [ ] Connection pooling on the C side: each PostgreSQL backend holds one persistent connection; pool of engines supports hundreds of connections.
- [ ] Backend authentication: engine checks peer UID via `SO_PEERCRED` matches postgres user.

### Task 4.4 — WAL integration

- [ ] Use PostgreSQL's Custom Resource Manager API (introduced in PG 15+) to register `turboquant` WAL records.
- [ ] WAL records capture the *intent* (vector data + tid), not the quantized output (which is deterministic given the rotator + codebook).
- [ ] On replay, replay records are forwarded to the engine.
- [ ] Standby replicas run their own local engine; each reapplies the raw vectors to produce bit-identical quantized indexes.

### Task 4.5 — Catalog integration

- [ ] Expose a system view `pg_turboquant_indexes` showing per-index stats: vector count, segment count, GPU memory, last seal time.
- [ ] `EXPLAIN` integration: custom `ExplainCustomPlanNode` shows estimated vs actual rows scanned in the quantized index.

**Acceptance criteria:**
- `TestPgTurboquantBuild`: `CREATE INDEX ... USING turboquant` on a 100k-row table completes in < 10s on H100.
- `TestPgTurboquantQuery`: `SELECT ... ORDER BY embedding <-> query LIMIT 10` returns top-10 with recall ≥ 0.95 vs brute force.
- `TestPgTurboquantCrashRecovery`: killing postgres during bulk insert, then restarting, results in a consistent index (all committed rows present, no uncommitted rows).
- Extension loads cleanly in PostgreSQL 16 and 17.

---

## 10. Component 5: Format Support — SafeTensors & GGUF (`turbodb-formats`)

**Purpose:** Import/export vectors and quantized indexes using industry-standard tensor file formats. No Python dependency for reading/writing — pure Go.

**Location:** `pkg/formats/safetensors/`, `pkg/formats/gguf/`

### Task 5.1 — SafeTensors reader

- [ ] Parse 8-byte little-endian length prefix → JSON header → tensor data section.
- [ ] Validate JSON header ≤ 100 MB (per SafeTensors spec).
- [ ] Support dtypes: `F32`, `F16`, `BF16`, `U8`, `I8`, `BOOL`. F16/BF16 converted to F32 on read.
- [ ] Zero-copy where possible via `mmap` (using `golang.org/x/exp/mmap`).
- [ ] `Iterator` API: `for name, tensor := range file.Iter()` — avoids loading the entire file at once.

### Task 5.2 — SafeTensors writer

- [ ] Build the JSON header with correct `data_offsets` (byte ranges within the tensor-data section).
- [ ] Stream tensor data directly to disk without full in-memory buffering (essential for multi-GB indexes).
- [ ] Custom metadata keys for TurboQuant:
  ```json
  {
    "__metadata__": {
      "turboquant_version": "0.1",
      "rotator_seed": "17384920123",
      "rotator_type": "hadamard",
      "codebook_id": "d1536_b4_lloyd_max_v1",
      "bit_width": "4",
      "variant": "mse"
    },
    "codes":  { "dtype": "U8", "shape": [1000000, 768], "data_offsets": [...] },
    "norms":  { "dtype": "F32", "shape": [1000000],   "data_offsets": [...] },
    "ids":    { "dtype": "I64", "shape": [1000000],   "data_offsets": [...] }
  }
  ```
- [ ] CLI: `turbodb-ctl export-safetensors --collection users --out users.safetensors`.

### Task 5.3 — GGUF reader

- [ ] Parse GGUF header: 4-byte magic `GGUF`, version (currently 3), tensor count, metadata count.
- [ ] Parse length-prefixed key-value metadata section.
- [ ] Parse tensor info array: name, ndims, dims, ggml_type, offset.
- [ ] Support loading weights (`F32`, `F16`, `Q4_0`, `Q4_K_M`, `Q8_0` at minimum) — dequantize to F32 when needed.
- [ ] Handle tensor data alignment per `general.alignment` metadata key.

### Task 5.4 — GGUF writer (weights and KV cache snapshots)

- [ ] Register two new ggml types in our own namespace (does not affect upstream llama.cpp): `GGML_TYPE_TURBOQUANT_MSE = 128`, `GGML_TYPE_TURBOQUANT_PROD = 129` (high IDs to avoid clashing with future upstream types).
- [ ] Document the layout in `docs/gguf-turboquant.md`:
  - Block size: 32 coordinates per block.
  - Block header: 4-byte `float16` norm + 2-byte seed offset into per-model rotator table.
  - Block body: packed `b`-bit codes.
- [ ] Writer produces llama.cpp-compatible GGUF files that our own reader can consume; interop with upstream llama.cpp is out of scope until our types are upstreamed.

### Task 5.5 — Import/export CLI

- [ ] `turbodb-ctl import --format safetensors --input embeddings.safetensors --collection docs`
- [ ] `turbodb-ctl import --format gguf --input model.gguf --tensor embedding.weight --collection vocab`
- [ ] `turbodb-ctl export --format safetensors --collection docs --output docs.safetensors`
- [ ] Streaming mode for files larger than RAM.
- [ ] Progress reporting via TTY-aware progress bars.

**Acceptance criteria:**
- `TestSafeTensorsRoundTrip`: write → read → compare tensors bit-identical.
- `TestSafeTensorsInterop`: files written by our Go library are readable by the Python `safetensors` library, and vice-versa, for all supported dtypes.
- `TestGGUFReadRealModel`: successfully load `Qwen3-0.6B-Q4_K_M.gguf` and extract embedding weights.
- `BenchmarkSafeTensorsRead_10GB`: full read in < 15s via mmap (bounded by disk I/O).

---

## 11. Component 6: KV Cache Integration (`turboquant-kv`)

**Purpose:** Provide TurboQuant as a KV cache compression backend for vLLM and SGLang. Ship as a pip-installable Python package backed by our Go library.

**Location:** `python/turboquant_kv/`

### Task 6.1 — C FFI surface

- [ ] Extend `libturboquant.so` with a KV-cache-specific API in `cuda/include/turboquant_kv.h`:
  ```c
  tq_status_t tq_kv_init(int d_head, int b_key, int b_val, tq_kv_handle_t* out);
  tq_status_t tq_kv_quantize_key(tq_kv_handle_t h, const void* k_d, int n_tokens,
                                 void* codes_out_d);
  tq_status_t tq_kv_quantize_val(...);
  tq_status_t tq_kv_attention(tq_kv_handle_t h,
                              const void* q_d, const void* k_codes_d,
                              const void* v_codes_d, int n_tokens,
                              void* output_d);
  ```
- [ ] The `tq_kv_attention` kernel performs quantized-key attention directly: it computes `softmax(Q · K_dequant^T / √d) · V_dequant` fused into a single kernel, following FlashAttention-style tiling.

### Task 6.2 — vLLM plugin

- [ ] `python/turboquant_kv/vllm_plugin.py` — register a new `kv_cache_dtype` option `"turboquant3"`, `"turboquant4"`, etc.
- [ ] Subclass `vllm.attention.AttentionBackend` with `TurboQuantBackend`.
- [ ] Reuse the approach from vLLM PR #38280 (Phase 1) — we are a direct continuation of that work, not a parallel effort.
- [ ] Support Paged KV Cache: quantize at page granularity (16 tokens per page by default).
- [ ] Register via vLLM's entry-point plugin system so users install and enable with two lines:
  ```python
  # pip install turboquant-kv
  from vllm import LLM
  llm = LLM(model="...", kv_cache_dtype="turboquant3")
  ```

### Task 6.3 — SGLang plugin

- [ ] `python/turboquant_kv/sglang_plugin.py` — SGLang's KV cache manager is more pluggable than vLLM's; implement the `BaseKVCache` interface.
- [ ] Mirror vLLM feature parity; share the same underlying `_ffi.py` CFFI layer.

### Task 6.4 — LMCache / Mooncake compression codec

- [ ] `python/turboquant_kv/lmcache_codec.py` — implement LMCache's codec interface for offloaded KV caches.
- [ ] Wire format uses SafeTensors with the TurboQuant metadata schema from Task 5.2, so offloaded caches are portable and inspectable.
- [ ] Benchmark against LMCache's default zstd codec for size and reconstruction latency.

### Task 6.5 — Accuracy and performance validation

- [ ] Needle-in-a-haystack test on Llama-3.1-8B-Instruct at context lengths 4k → 104k, matching the paper's methodology.
- [ ] LongBench evaluation: run the full suite, reproduce the paper's Table 1 numbers within statistical noise.
- [ ] Throughput: measure tokens/sec on H100 with batch sizes 1, 8, 32, 64 at contexts 4k, 32k, 128k. Target ≥ 20% throughput improvement at batch 16 over FP8 baseline (the vLLM PR showed ~21%).

**Acceptance criteria:**
- `pytest python/turboquant_kv/tests/test_accuracy.py` — LongBench score within 1% of full-precision baseline at `b=3.5`.
- `pytest python/turboquant_kv/tests/test_throughput.py` — tokens/sec ≥ FP8 baseline, memory ≤ 5× smaller.
- Package installs cleanly in fresh venvs on Python 3.10, 3.11, 3.12.

---

## 12. Component 7: CDC & Replication (`turbodb-sync`)

**Purpose:** Keep `turbodb-engine` in sync with PostgreSQL as the source of truth for raw vectors. Uses PostgreSQL logical replication (pgoutput or wal2json plugin) consumed by a Go service.

**Location:** `cmd/turbodb-sync/`, `pkg/replication/`

### Task 7.1 — Logical replication consumer

- [ ] Use `jackc/pglogrepl` to consume a PostgreSQL replication slot.
- [ ] Parse `Insert`, `Update`, `Delete` messages; filter for configured tables.
- [ ] Schema evolution: resolve column names via publication metadata.
- [ ] Checkpoint LSN to local Badger store so restarts resume from last-committed position.

### Task 7.2 — Event transformer

- [ ] Config file (`sync.yaml`) maps PostgreSQL tables to TurboDB collections:
  ```yaml
  tables:
    - postgres: public.documents
      engine:   docs
      columns:
        id:        doc_id      # -> collection primary key
        embedding: vector      # -> vector column
      filter:    "deleted_at IS NULL"
  ```
- [ ] Applies filters; skips rows that don't match (`deleted_at IS NOT NULL` etc.).

### Task 7.3 — Engine writer

- [ ] Batched gRPC calls to the engine's `InsertBatch`/`Delete` RPCs.
- [ ] Backpressure: if the engine is overloaded, pause WAL consumption (don't ack LSN).
- [ ] Retry with exponential backoff + jitter on transient errors.
- [ ] Circuit breaker: after N consecutive failures, stop and emit critical alert.

### Task 7.4 — Reconciliation job

- [ ] Periodic (default: hourly) full-table diff:
  1. Scan PostgreSQL source table in primary-key order.
  2. Scan engine collection in primary-key order.
  3. For any mismatch, emit repair events.
- [ ] Run as a separate `turbodb-sync reconcile` subcommand.
- [ ] Expose metrics: `turbodb_sync_reconcile_discrepancies_total`, `turbodb_sync_reconcile_last_run_seconds`.

### Task 7.5 — Kafka mode (optional, higher scale)

- [ ] Alternative transport: PostgreSQL → Debezium → Kafka → `turbodb-sync` consumer.
- [ ] Same transformer and writer code paths; only the input source changes.
- [ ] Useful for fan-out (one source of truth, many downstream engines).

**Acceptance criteria:**
- `TestSyncEndToEnd`: insert 10k rows into Postgres, `turbodb-sync` delivers them to the engine within 5s, reconciliation finds zero discrepancies.
- `TestSyncDurability`: kill `turbodb-sync` mid-stream; on restart it resumes from the correct LSN with no data loss and no duplication.

---

## 13. Component 8: Control Plane & Operations (`turbodb-ctl`)

**Purpose:** CLI and admin API for day-two operations.

**Location:** `cmd/turbodb-ctl/`

### Task 8.1 — CLI structure (Cobra-based)

```
turbodb-ctl
├── collection
│   ├── create    (--name, --dim, --bits, --metric)
│   ├── list
│   ├── describe  <name>
│   ├── drop      <name>
│   └── flush     <name>
├── index
│   ├── build-stats <collection>
│   └── compact     <collection>
├── import        --format (safetensors|gguf|parquet) --input <path> --collection <name>
├── export        --format safetensors --collection <name> --output <path>
├── benchmark
│   ├── throughput --collection <name> --n <count>
│   └── recall     --collection <name> --ground-truth <path>
├── sync
│   ├── status
│   ├── reconcile <collection>
│   └── repair    <collection>
├── admin
│   ├── rotator-regenerate <collection>   # SECURITY SENSITIVE — explicit --confirm
│   ├── codebook-upgrade    <collection> --from <version> --to <version>
│   └── gpu-info
└── version
```

### Task 8.2 — Admin HTTP API

- [ ] Simple HTTP/JSON API on a separate port (default 8080) for automation tooling.
- [ ] `/healthz`, `/readyz`, `/metrics` (Prometheus).
- [ ] `/api/v1/collections` (GET list, POST create), `/api/v1/collections/{name}` (GET describe, DELETE drop).
- [ ] mTLS required for write endpoints in production mode.

### Task 8.3 — Benchmarking harness

- [ ] `cmd/turbodb-bench/` — reproducible benchmark driver.
- [ ] Datasets: GloVe 200d, DBpedia 1536d, DBpedia 3072d (matching the paper).
- [ ] Metrics: throughput (QPS), latency percentiles, recall@k for k ∈ {1, 10, 100}, indexing time, memory footprint.
- [ ] Output JSON results consumable by Grafana or a comparison spreadsheet.

**Acceptance criteria:**
- CLI has bash/zsh completion.
- `turbodb-ctl admin rotator-regenerate` refuses to run without `--confirm "i understand this will invalidate all indexes"`.
- Benchmarks are reproducible: same seed → identical results within 1% across runs.

---

## 14. Testing Strategy

### Test taxonomy

| Level | Location | What it tests | CI runs on |
|---|---|---|---|
| Unit | `*_test.go` alongside source | Pure functions, data structures, algorithms | Every PR |
| Integration | `test/integration/` | Multi-package interactions (no GPU, no Postgres) | Every PR |
| GPU integration | `test/integration/` with `//go:build gpu` | CUDA kernels, GPU fallback paths | Nightly (self-hosted GPU runner) |
| Postgres integration | `test/integration/` with `//go:build postgres` | Extension load, index build, query | Nightly (testcontainers-go) |
| End-to-end | `test/e2e/` | Full stack: Postgres + engine + sync + client | Nightly |
| Benchmark | `cmd/turbodb-bench/` | Performance regressions | Weekly + on release |
| Fuzz | `*_fuzz_test.go` | Parser inputs (SafeTensors, GGUF, IPC protocol) | Weekly |

### Coverage targets

- **Unit + integration:** ≥ 80% statement coverage for `pkg/`.
- **GPU integration:** every public CUDA kernel has a parity test against CPU reference.
- **E2E:** one golden-path test per deployment shape (embedded, standalone, hybrid, KV-only).

### Continuous integration

- **Main CI (every PR):** lint, format check, `go test ./...` (no GPU tag), protobuf breaking-change check, documentation build.
- **GPU CI (nightly):** `go test -tags gpu ./...` on self-hosted A100 runner.
- **Release CI:** runs all of the above + benchmarks; produces container images, binaries, and Python wheel.

### Property-based testing

- Use `github.com/flyingmutant/rapid` for properties like:
  - `QuantizeDequantize(x)` returns a vector within paper-bounded distortion.
  - `Rotator.Apply` preserves L2 norm.
  - IPC protocol framing: `Encode → Decode = identity`.

---

## 15. Security Model

### Threat model (what we defend against)

1. **Malicious input files** — a user uploads a crafted SafeTensors or GGUF file.
2. **Network attacker** between client and engine.
3. **Compromised PostgreSQL backend** trying to abuse the engine's IPC socket.
4. **Credential theft** — leaked mTLS private keys.

### Mitigations

- **File parsing:** fuzz all parsers (SafeTensors JSON header, GGUF metadata, IPC frames). Set hard limits: header ≤ 100MB for SafeTensors, tensor count ≤ 1M, dim ≤ 65,536. Reject files with oversized or malformed fields before any allocation.
- **No pickle, ever.** We don't touch Python pickle. If a user supplies a `.bin` or `.ckpt`, the CLI refuses and points them to convert via `safetensors`.
- **IPC socket permissions:** Unix socket is `0600`, owned by the postgres user. Peer credential check via `SO_PEERCRED` rejects connections from other UIDs.
- **gRPC:** mTLS required in production mode; plaintext only allowed if `--dev` flag is set and emits a startup warning.
- **Secrets management:** rotator seeds and codebook IDs are not secret (they're deterministic defaults for a given collection config). Private keys for mTLS live on disk with `0600` perms, optionally loaded from HashiCorp Vault.
- **Resource limits:** each gRPC request has a max message size (default 64MB). Per-client rate limiting via `golang.org/x/time/rate`.
- **Dependency hygiene:** `go.mod` pinned, `govulncheck` in CI, Dependabot alerts reviewed weekly.

### What we don't defend against

- Side-channel attacks on GPU memory (shared-tenant GPUs are explicitly out of scope).
- Adversarial quantization attacks (see recent GGUF research) — we note this in documentation and recommend production deployments validate model behavior post-quantization.

---

## 16. Deployment

### Container images

- `turbodb/engine:<version>` — base image is `nvidia/cuda:12.4-runtime-ubuntu22.04`, statically-linked Go binary plus `libturboquant_cuda.so`.
- `turbodb/sync:<version>` — Alpine-based, pure Go, no CUDA dependency.
- `turbodb/postgres:<version>` — extends `postgres:16`, includes `pg_turboquant.so`.

### Kubernetes (Helm chart at `deploy/helm/turbodb/`)

- `StatefulSet` for `turbodb-engine` — GPU node selector (`nvidia.com/gpu: 1`), persistent volume for segments and WAL.
- `Deployment` for `turbodb-sync` — stateless (LSN checkpointed in Postgres itself via a dedicated table).
- `Service` + `Ingress` for gRPC + admin HTTP.
- `ServiceMonitor` CRDs for Prometheus Operator.
- `ConfigMap` for engine config; `Secret` for mTLS material.

### Single-machine / bare-metal

- `deploy/systemd/` — unit files for `turbodb-engine.service` and `turbodb-sync.service`.
- `/etc/turbodb/engine.yaml` — config file.
- Standard `/var/lib/turbodb/` data directory.

### Resource sizing guidelines (documented in `docs/operations.md`)

- **Engine:** 1 GPU (A100 40GB min, H100 80GB recommended), 16 CPU cores, 128GB RAM, NVMe SSD with 1TB+ for WAL and cold segments.
- **Sync:** 2 CPU cores, 4GB RAM, small local disk for checkpoint store.
- **Postgres + extension:** standard Postgres sizing, no GPU required (GPU work happens in the engine).

---

## 17. Phased Delivery & Milestones

### Phase 0 — Foundations (weeks 1–2)

Set up repo, tooling, CI. No product code.

- [x] Initialize Go module, lint config, protobuf build.
- [ ] Skeleton CI: lint + unit tests on every PR.
- [x] Protobuf definitions for all services (even if unimplemented).
- [x] CONTRIBUTING.md, code-of-conduct, issue templates.

**Exit:** `make build test lint` passes on an empty repo. ✅ Verified.

### Phase 1 — CPU-only TurboQuant (weeks 3–6)

All of Component 1. No GPU, no services yet. This is the algorithmic correctness foundation that everything else validates against.

- [x] All 5 tasks complete (1.1–1.5).
- [x] 61 tests passing, 0 lint issues.
- [x] No data races under `go test -race ./...`.
- [x] `make build test lint` passes.

**Exit:** `pkg/quantizer` passes all accuracy tests against paper's Table 1. ✅ Verified.

### Phase 2 — CUDA kernel layer (weeks 7–12)

All of Component 2. Kernels + cgo bindings. Still no services.

- [x] All 6 tasks complete (2.1–2.6).
- [x] CUDA kernels: context, FWHT, quantize/dequantize, QJL, brute-force search.
- [x] Go cgo bindings with `cuda` build tag + CPU fallback stub.
- [x] GoogleTest parity tests (FWHT round-trip, norm preservation, determinism, quantize/dequantize).
- [x] CUDA Makefile + root Makefile integration (`make cuda`, `make cuda-test`, `make build-cuda`).
- [x] `go build` (no tag) passes on macOS. 65 tests passing.
- [ ] GPU parity tests pending (requires CUDA machine for `make cuda-test` and `go test -tags cuda`).

**Exit:** parity tests pass; `cmd/turbodb-bench` shows GPU ≥ 1000× faster than CPU reference for quantize throughput.

### Phase 3 — Standalone engine MVP (weeks 13–18)

Component 3 minus replication and advanced planner features. Single-node, in-memory + WAL, CRUD + search.

**Exit:** engine serves 100k-vector collection via gRPC, recall ≥ 0.95, p99 < 20ms, survives crash-recover.

### Phase 4 — Format support + KV cache (weeks 19–24, parallel tracks)

- Track A: Component 5 (SafeTensors, GGUF). Independent, can ship without engine changes.
- Track B: Component 6 (vLLM/SGLang plugins). Depends on Phase 2 (CUDA layer).

**Exit:** users can `pip install turboquant-kv` and enable TurboQuant in vLLM; `turbodb-ctl import --format safetensors` works end-to-end.

### Phase 5 — PostgreSQL extension (weeks 25–32)

Component 4. Highest-risk phase due to C code and Postgres internals.

**Exit:** `CREATE EXTENSION pg_turboquant; CREATE INDEX ... USING turboquant` works; passes Postgres regression-style tests.

### Phase 6 — Replication, CDC, production hardening (weeks 33–40)

Components 7 and 8, plus production concerns.

**Exit:** hybrid deployment passes 24-hour soak test with injected faults (engine crash, network partition, Postgres failover).

### Phase 7 — GA readiness (weeks 41–44)

Documentation, operational runbooks, performance targets all met, security review passed.

**Exit:** GA release.

---

## 18. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| CUDA kernel performance below target | Medium | High | Prototype FWHT kernel in week 1 of Phase 2 to de-risk; have cuBLAS GEMM fallback ready. |
| PostgreSQL extension API churn between versions | Low | Medium | Target PG 16/17 LTS; use `#if PG_VERSION_NUM` guards. Watch pgvector and PG-Strom changelogs. |
| vLLM internal API changes break plugin | High | Medium | Pin to known-good vLLM versions; coordinate with vLLM PR #38280 maintainers. Run plugin CI against multiple vLLM versions. |
| Go/cgo performance overhead at hot path | Low | Medium | Batch all cgo calls (never call per-vector). Benchmark cgo boundary in Phase 2; if overhead exceeds 2%, revisit. |
| GPU memory pressure for billion-scale indexes | Medium | Medium | Tiered storage (GPU → pinned host → NVMe) from day one. Document capacity planning. |
| SafeTensors / GGUF format evolves | Low | Low | Versioned readers; add new versions without breaking old ones. Contribute test cases upstream. |
| Regulatory / compliance requirements for AI systems | Medium | Low-Medium | Design supports full audit log of all mutations. Segment data deletion honors `DROP COLLECTION` within SLA. |

---

## 19. Success Metrics

### Correctness

- MSE distortion within 10% of Theorem 1 bounds across all `(d, b)` pairs.
- Inner-product estimates unbiased (Welch's t-test p > 0.05 on 10k samples).
- Recall@10 ≥ 0.95 on DBpedia-1536 at `b=4`.

### Performance

- Indexing throughput: ≥ 50M vectors/sec on H100 at `d=1536, b=4`.
- Search throughput: ≥ 200k QPS at top-10 over 1M vectors, single H100.
- KV cache: ≥ 5× memory reduction at `b=3`, LongBench score within 1% of FP16.

### Operability

- Single-binary install for engine.
- Crash-recovery time < 30s for 10M-vector collection.
- Full documentation builds on every PR.
- 80%+ test coverage on `pkg/`.

### Adoption signals

- `pip install turboquant-kv` works in a fresh venv.
- Non-trivial contributions from outside the core team within 6 months of GA.
- At least one upstream PR contribution back to vLLM, cuVS, or llama.cpp as a result of this work.

---

## Appendix A — Reference Links

- TurboQuant paper: https://arxiv.org/abs/2504.19874
- vLLM TurboQuant PR: https://github.com/vllm-project/vllm/pull/38280
- pgvector: https://github.com/pgvector/pgvector
- PG-Strom: https://heterodb.github.io/pg-strom/
- cuVS: https://rapids.ai/cuvs/
- SafeTensors spec: https://huggingface.co/docs/safetensors
- GGUF spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

## Appendix B — Glossary

- **FWHT** — Fast Walsh–Hadamard Transform; `O(d log d)` algorithm for applying a Hadamard matrix.
- **QJL** — Quantized Johnson–Lindenstrauss transform; the 1-bit sketching step in TurboQuant's inner-product variant.
- **Lloyd-Max** — iterative algorithm for optimal scalar quantization of a known probability distribution.
- **CAGRA** — GPU-native graph ANN index from NVIDIA's cuVS library.
- **cgo** — Go's mechanism for calling C code.
- **LSN** — PostgreSQL Log Sequence Number, a monotonic position in the WAL.
- **AM** — Access Method, PostgreSQL's pluggable-index abstraction (`IndexAmRoutine`).
