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
CUDA Kernel Layer          | cuda/, internal/cuda/ | CUDA/Go  | Phase 2
Standalone GPU Engine      | cmd/turbodb-engine/   | Go       | Phase 3
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

**Next:** Phase 2 — CUDA kernel layer.

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
