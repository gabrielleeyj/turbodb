# Contributing to TurboDB

Thank you for your interest in contributing to TurboDB.

## Getting Started

### Prerequisites

- Go 1.25.6+
- Make
- golangci-lint (recommended)
- buf (for protobuf changes)
- CUDA 12.x toolkit (for GPU components only)

### Build

```bash
make build
```

### Test

```bash
make test
```

### Lint

```bash
make lint
```

## Development Workflow

1. Fork the repository and create a feature branch from `main`.
2. Write tests first (TDD). See the testing guidelines below.
3. Make your changes.
4. Run `make all` (builds, tests, lints).
5. Open a pull request.

## Commit Messages

Use conventional commits:

```
<type>: <description>

<optional body>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`

## Code Style

- Run `gofmt` and `goimports` on all Go files.
- Follow idiomatic Go patterns (accept interfaces, return structs).
- Keep functions under 50 lines, files under 800 lines.
- Handle all errors explicitly with context wrapping.
- Never mutate shared state; prefer immutable data.

## Testing

- All new code must have tests.
- Target 80%+ statement coverage for `pkg/`.
- Use table-driven tests.
- Run with `-race` flag.
- GPU tests use `//go:build gpu` tags.
- PostgreSQL tests use `//go:build postgres` tags.

### Package-Specific Testing

**`pkg/codebook`**

```bash
# Run all codebook tests
go test -race -v ./pkg/codebook/

# Regenerate precomputed codebooks (slow, ~minutes)
GENERATE_CODEBOOKS=1 go test -v -run TestGeneratePrecomputed ./pkg/codebook/
```

The codebook tests cover density integration, Lloyd-Max convergence, centroid symmetry, distortion monotonicity, and loading all 56 precomputed codebooks.

**`pkg/rotation`**

```bash
go test -race -v ./pkg/rotation/

# Benchmark rotation performance
go test -bench BenchmarkHadamardRotator -benchmem ./pkg/rotation/
```

Tests verify norm preservation, round-trip accuracy, determinism, and serialization.

**`pkg/quantizer`**

```bash
go test -race -v ./pkg/quantizer/

# Benchmark batch quantization
go test -bench BenchmarkBatchQuantize -benchmem ./pkg/quantizer/
```

Tests cover MSE round-trip accuracy, bit packing exactness, ProdQuantizer unbiasedness and variance, batch determinism, streaming, and data race detection.

## Protobuf Changes

- Edit `.proto` files in `api/v1/`.
- Run `buf lint` before committing.
- Run `buf breaking --against .git#branch=main` to check for breaking changes.
- Run `make proto` to regenerate Go code.

## Security

- Never commit secrets, API keys, or credentials.
- All file parsers must be fuzz-tested.
- Validate all external input at system boundaries.
- Report security vulnerabilities privately (do not open public issues).
