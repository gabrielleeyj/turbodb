# turbodb

Building a TurboQuant Database Extension - Experimental

Recommended Hardware: NVIDIA H100 / A100 (CUDA 12.x), x86_64 Linux  
Architecture: Hybrid PostgreSQL extension + standalone GPU engine + LLM framework plugins  
Reference algorithm: TurboQuant [arXiv:2504.19874v1, ICLR 2026](https://arxiv.org/html/2504.19874v1)

# Architecture Overview

TurboDB is a GPU-accelerated database system built around the TurboQuant vector
quantization algorithm (arXiv:2504.19874v1). It provides two primary capabilities:

1. **Vector similarity search** with 8x compression at 4-bit, near-optimal recall.
2. **KV cache quantization** for LLM serving (vLLM/SGLang), reducing memory 5-6x.

## Components

Scope of work planned in phases.

```
Component                  | Location              | Language | Status
---------------------------|-----------------------|----------|--------
Core TurboQuant Library    | pkg/                  | Go       | Done
CUDA Kernel Layer          | cuda/, internal/cuda/ | CUDA/Go  | Partially Complete
Standalone GPU Engine      | cmd/turbodb-engine/   | Go       | Phase 3 Complete
PostgreSQL Extension       | postgres/             | C/Go     | Phase 5
Format Support             | pkg/formats/          | Go       | Phase 4
KV Cache Plugin            | python/               | Python   | Phase 4
CDC & Replication          | cmd/turbodb-sync/     | Go       | Phase 6
Control Plane CLI          | cmd/turbodb-ctl/      | Go       | Phase 6
```

## Language Boundaries

- **Go**: Control plane, services, orchestration, IPC, APIs, lifecycle.
- **CUDA C++**: Hot-path GPU kernels (FWHT, quantize, QJL, search).
- **C**: PostgreSQL extension shim (minimal, IPCs to Go daemon).
- **Python**: vLLM/SGLang plugin wrappers over CFFI or gRPC.

Design rule: every non-Go component is wrapped in a Go interface at its narrowest
point. Tests at the interface can run without a GPU.

## Deployment Shapes

1. **Embedded in PostgreSQL only** — pg_turboquant extension, no standalone engine.
2. **Standalone engine only** — turbodb-engine serving gRPC.
3. **Hybrid** (recommended) — PostgreSQL as source of truth, engine serves quantized index, sync keeps them consistent.
4. **KV cache only** — turboquant-kv plugin in vLLM/SGLang, no Postgres, no engine.

## Data Flow

```
Raw vectors → Rotation (FWHT/QR) → Scalar quantize (codebook lookup) → Packed bit codes
                                                                          ↓
Query vector → Rotation → Quantize → Inner-product estimate → Top-K merge → Results
```

For the PROD variant, an additional QJL sketch of the residual is stored alongside
the MSE codes to provide unbiased inner-product estimates.

## Current Status

**Phase 1 complete** — CPU-only TurboQuant reference implementation.

- 61 tests passing, 0 lint issues, no data races under `-race`
- `pkg/codebook`: 56 precomputed codebooks for all standard (d, b) pairs, Lloyd-Max solver with symmetry exploitation + overrelaxation
- `pkg/rotation`: Randomized FWHT via HadamardRotator, O(d log d), ~9µs/vector at d=1536
- `pkg/quantizer`: MSEQuantizer (Algorithm 1), ProdQuantizer (Algorithm 2), QJL sketch, batch/streaming APIs

**Phase 2 partially complete** — CUDA kernel layer code written, GPU parity testing pending.

- CUDA kernels: `cuda/src/{fwht,quantize,qjl,search,codebook,context}.cu` with C ABI in `cuda/include/turboquant.h`
- Go cgo bindings: `internal/cuda/` builds cleanly with and without the `cuda` build tag
- 65 Go tests passing on CPU; GoogleTest suites for FWHT and quantize round-trips written
- **Pending:** GPU parity validation requires a CUDA machine / CI runner

**Phase 3 complete** — Standalone GPU engine MVP.

- Task 3.1 — segment architecture: `pkg/index/` provides Segment / GrowingSegment / SealedSegment / Collection with a background sealer, TombstoneLog, and a CRC32C-framed segment file format.
- Task 3.2 — write-ahead log: `pkg/wal/` implements length-prefixed CRC32C records, typed payloads (Insert / Delete / SegmentSealed / Checkpoint), file rotation, FsyncEveryWrite + FsyncGroupCommit, LSN persistence across reopen, and tail-corruption-tolerant `Iterate()` recovery.
- Task 3.3 — gRPC API: `cmd/turbodb-engine/` boots a gRPC listener; `internal/engine/` orchestrates collections, the WAL, and replay-on-start.
- Task 3.4 — query planner: `pkg/search/` exposes `Options`/`Plan`/`Planner` on top of `*index.Collection` with oversearch, optional rerank, and per-call telemetry. `internal/engine.Search` returns a `(results, plan, error)` tuple; gRPC `SearchRequest` forwards `top_k`, `rerank`, `ef_search`, and `exact`.
- Task 3.5 — memory management: `pkg/memory/` provides a semaphore-backed `Budget` (admission control + accounting) and a sealed-segment byte estimator. `EngineConfig.MemoryBudgetBytes` (0 = unlimited) flows into every collection; `CollectionStats.PinnedBytes` and `Engine.MemoryStats()` surface usage. Spill-to-NVMe and async prefetch deferred until on-disk segment loading lands.
- Task 3.6 — observability: `pkg/telemetry/` exposes a Prometheus `Metrics` bundle (`turbodb_search_latency_seconds`, `_insert_throughput_vectors_total`, `_segments_sealed_total`, `_segments_active`, `_host_memory_bytes`, `_gpu_memory_bytes`, `_wal_fsync_latency_seconds`), an OTel tracer scope, and a `slog` logger factory. `Engine.AttachMetrics` resolves the engine ↔ StatsSource cycle via an `atomic.Pointer`. `cmd/turbodb-engine` exposes `/metrics` and `/healthz` on `--metrics-listen :9090` and accepts `--log-format json|text` and `--log-level`.

**Phase 3 exit-criteria benchmark** — `cmd/turbodb-bench` loads N synthetic vectors, brute-forces ground truth, and reports recall@k plus p50/p95/p99 latency. First run on Apple M3 Pro (CPU only):

| Criterion (SCOPE §17) | Target | Result |
|---|---|---|
| 100k-vector collection served | ✓ | 100k, 4-bit MSE, 15.6 MiB pinned |
| Recall@10 | ≥ 0.95 | **1.0000** mean |
| Search p99 | < 20 ms | **31 ms** (CPU MVP) |
| Crash-recover | survives | recall preserved on close+reopen |

Recall and recovery are met. The 20 ms p99 SLO is not reachable on a single core: the hot path is dequant + inner-product over the entire sealed segment, bandwidth-bound at this scale. Closing this gap is GPU-dispatch work — the kernels in `cuda/` are not yet wired into `pkg/index/sealed.go` Search.

```bash
go run ./cmd/turbodb-bench/ -vectors 100000 -queries 200 -dim 256 -bit-width 4 -top-k 10 -insert-workers 8 -oversearch 2.0
go run ./cmd/turbodb-bench/ -vectors 1000  -queries 30  -crash-recover
```

**Next:** wire CUDA dispatch into the sealed-segment search path, then Phase 4 (formats + Postgres FDW skeleton).

## Quickstart

```bash
# Prerequisites: Go 1.23+
git clone https://github.com/gabrielleeyj/turbodb.git
cd turbodb

# Build and test
make build test lint

# Run tests with race detection
go test -race ./...
```

### Usage Example

```go
package main

import (
    "fmt"
    "math/rand/v2"

    "github.com/gabrielleeyj/turbodb/pkg/codebook"
    "github.com/gabrielleeyj/turbodb/pkg/quantizer"
    "github.com/gabrielleeyj/turbodb/pkg/rotation"
)

func main() {
    dim, bitWidth := 256, 4

    // 1. Load codebook and create rotator.
    rot, _ := rotation.NewHadamardRotator(dim, 42)
    cb, _ := codebook.Load(dim, bitWidth-1) // MSE uses b-1 bits

    // 2. Build quantizers.
    mseQ, _ := quantizer.NewMSEQuantizer(dim, bitWidth-1, rot, cb)
    qjl, _ := quantizer.NewQJLSketch(dim, dim, 99)
    pq, _ := quantizer.NewProdQuantizer(dim, bitWidth, mseQ, qjl)

    // 3. Quantize a vector.
    rng := rand.New(rand.NewPCG(1, 2))
    x := make([]float32, dim)
    for i := range x {
        x[i] = rng.Float32() - 0.5
    }
    code, _ := pq.Quantize(x)

    // 4. Estimate inner product with a query.
    y := make([]float32, dim)
    for i := range y {
        y[i] = rng.Float32() - 0.5
    }
    ip, _ := pq.EstimateInnerProduct(y, code)
    fmt.Printf("Estimated inner product: %f\n", ip)

    // 5. Batch quantize many vectors.
    xs := make([][]float32, 1000)
    for i := range xs {
        v := make([]float32, dim)
        for j := range v {
            v[j] = rng.Float32() - 0.5
        }
        xs[i] = v
    }
    codes, _ := quantizer.BatchQuantize(mseQ, xs)
    fmt.Printf("Batch quantized %d vectors\n", len(codes))
}
```
