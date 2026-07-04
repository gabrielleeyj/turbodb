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
PostgreSQL Extension       | postgres/             | C/Go     | Phase 5 (compile-verified; GPU acceptance pending)
Format Support             | pkg/formats/          | Go       | Phase 4 Complete
KV Cache Plugin            | python/               | Python   | Phase 4 (scaffold; GPU kernel pending)
CDC & Replication          | cmd/turbodb-sync/     | Go       | Phase 6 (Tasks 7.1-7.3 done; reconcile pending)
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

**Phase 4 complete** — Format support + KV-cache scaffold.

- Track A — Component 5 (formats), pure Go: `pkg/formats/safetensors` (mmap reader + streaming writer, F16/BF16 conversion, TurboQuant `__metadata__` schema) and `pkg/formats/gguf` (GGUF v3 reader/writer, dequant of F32/F16/Q8_0/Q4_0/Q4_1, custom TurboQuant ggml types 128/129 — see `docs/gguf-turboquant.md`). `turbodb-ctl import|export|inspect|convert` wires these to the engine via `internal/ioformats`. 80%+ coverage across all three.
- Track B — Component 6 (KV cache): `cuda/include/turboquant_kv.h` defines the KV C ABI; `python/turboquant_kv` is a pip-installable package (imports without CUDA) with `KVConfig`/dtype parsing, a ctypes FFI loader, and vLLM / SGLang / LMCache plugin skeletons (21 tests pass off-GPU). The fused `tq_kv_attention` kernel and accuracy/throughput validation (Task 6.5) are GPU-blocked and exercised in GPU CI.

**Phase 5 substantially complete** — `pg_turboquant` PostgreSQL extension.

- IPC protocol (Task 4.3): `internal/pgproto` (Go server) + `postgres/turbodb_ipc.{c,h}` (C client) speak a length-prefixed binary frame protocol; wire compatibility is proven by a cross-language test that compiles the C client and runs it against the Go engine (`internal/pgipc`).
- C extension (Tasks 4.1, 4.2, 4.5): `postgres/pg_turboquant.c` implements the `turboquant` `IndexAmRoutine` (build via heap scan → IPC, insert, scan/search, reloptions, costestimate) plus DDL, PGXS Makefile, and the `pg_turboquant_indexes` catalog view. Compiles clean and links against PostgreSQL 18.
- Deferred (integration/GPU-blocked): Custom WAL Resource Manager (Task 4.4) and the `CREATE INDEX`-on-100k acceptance tests, which need a live cluster with pgvector and the GPU engine.

**Phase 6 started** — CDC & replication pipeline core (Component 7).

- `pkg/replication`: sync.yaml config with strict validation (Task 7.2), a compiled filter subset (`IS NULL` / `IS NOT NULL` / `=` / `!=` / `AND`), event transformer with soft-delete semantics (an update that stops matching the filter becomes an engine delete), CRC32C-protected atomic file LSN checkpoint, and a batching engine writer with exponential-backoff retry + jitter and a consecutive-failure circuit breaker (Task 7.3). The `Sync` loop wires source → transform → write → checkpoint and only acks LSNs after successful flushes. 89% coverage, race-clean.
- Task 7.1 — logical replication consumer: `PgSource` consumes a replication slot via `jackc/pglogrepl` (pgoutput), resolves column names from Relation messages (schema evolution), buffers events per transaction and stamps them with the transaction end LSN so a checkpointed restart never replays or loses a transaction. Delivery is at-least-once with idempotent engine writes; PostgreSQL WAL is only released past LSNs that were flushed to the engine AND checkpointed (`LSNAcker`). Integration tests (`TestPgSourceEndToEnd`, `TestPgSourceDurabilityAcrossRestart`) run against a live `pgvector/pgvector:pg17` container when `TURBODB_TEST_PG_DSN` is set and skip otherwise.
- `cmd/turbodb-sync run` wires PgSource -> transformer -> gRPC engine writer -> checkpoint; `check-config` validates a sync.yaml. Verified live end-to-end: Postgres inserts + a soft-delete replicated into a running `turbodb-engine`, with search correctly excluding the tombstoned row.

**Next:** Task 7.4 (reconciliation job), Component 8 control-plane expansion, and wiring CUDA dispatch into the sealed-segment search path (closes the Phase 3 p99 SLO and the GPU-blocked acceptance tests above).

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
