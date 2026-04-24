.PHONY: build test lint fmt vet clean proto all cuda cuda-test cuda-clean

# Default target
all: build test lint

# Build all Go binaries
BINARIES := turbodb-engine turbodb-ctl turbodb-sync turbodb-bench
BUILD_DIR := bin

build:
	@echo "==> Building all binaries..."
	@mkdir -p $(BUILD_DIR)
	@for bin in $(BINARIES); do \
		echo "    $$bin"; \
		go build -o $(BUILD_DIR)/$$bin ./cmd/$$bin; \
	done
	@echo "==> Done."

# Run all tests with race detection
test:
	@echo "==> Running tests..."
	go test -race -count=1 ./...
	@echo "==> Done."

# Run tests with coverage
test-cover:
	@echo "==> Running tests with coverage..."
	go test -race -coverprofile=coverage.out ./...
	go tool cover -func=coverage.out
	@echo "==> Done."

# Lint using golangci-lint
lint:
	@echo "==> Running linters..."
	@if command -v golangci-lint >/dev/null 2>&1; then \
		golangci-lint run ./...; \
	else \
		echo "    golangci-lint not found, running go vet only"; \
		go vet ./...; \
	fi
	@echo "==> Done."

# Format code
fmt:
	@echo "==> Formatting..."
	gofmt -w .
	@if command -v goimports >/dev/null 2>&1; then \
		goimports -w .; \
	fi
	@echo "==> Done."

# Run go vet
vet:
	go vet ./...

# Generate protobuf code (requires buf)
proto:
	@echo "==> Generating protobuf code..."
	@if command -v buf >/dev/null 2>&1; then \
		cd api && buf generate; \
	else \
		echo "    buf not installed, skipping proto generation"; \
	fi
	@echo "==> Done."

# Build CUDA shared library (requires CUDA toolkit)
cuda:
	@echo "==> Building CUDA kernels..."
	$(MAKE) -C cuda
	@echo "==> Done."

# Run CUDA GoogleTest tests (requires GPU)
cuda-test: cuda
	@echo "==> Running CUDA tests..."
	$(MAKE) -C cuda test
	@echo "==> Done."

# Build Go binaries with CUDA support
build-cuda: cuda
	@echo "==> Building all binaries (with CUDA)..."
	@mkdir -p $(BUILD_DIR)
	@for bin in $(BINARIES); do \
		echo "    $$bin (cuda)"; \
		CGO_LDFLAGS="-L$(CURDIR)/cuda/lib -lturboquant_cuda" \
		go build -tags cuda -o $(BUILD_DIR)/$$bin ./cmd/$$bin; \
	done
	@echo "==> Done."

# Run Go tests with CUDA support
test-cuda: cuda
	@echo "==> Running Go tests (with CUDA)..."
	CGO_LDFLAGS="-L$(CURDIR)/cuda/lib -lturboquant_cuda" \
	LD_LIBRARY_PATH=$(CURDIR)/cuda/lib:$(LD_LIBRARY_PATH) \
	go test -tags cuda -race -count=1 ./...
	@echo "==> Done."

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	rm -f coverage.out

cuda-clean:
	$(MAKE) -C cuda clean
