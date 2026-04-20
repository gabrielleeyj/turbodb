# turbodb

Building a TurboQuant Database Extension

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
Core TurboQuant Library    | pkg/                  | Go       | Phase 1
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
