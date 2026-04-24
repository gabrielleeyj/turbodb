package cuda

// Context wraps a tq_context_t handle for GPU operations.
// Each Context owns a dedicated CUDA stream and must be closed when done.
//
// A Context is NOT safe for concurrent use. Use one per goroutine/stream.
type Context interface {
	// Close destroys the context and frees associated GPU resources.
	Close()

	// Sync blocks until all enqueued GPU operations complete.
	Sync() error

	// LastError returns a human-readable description of the last CUDA error.
	LastError() string

	// DeviceInfo returns GPU memory and compute capability information.
	DeviceInfo() (DeviceInfo, error)

	// CreateCodebook uploads centroids to the GPU and returns a handle.
	CreateCodebook(centroids []float32, bitWidth int) (Codebook, error)

	// CreateRotator creates a Hadamard rotator on the GPU.
	CreateRotator(dim int, seed uint64) (Rotator, error)

	// FWHTBatch applies batched randomized Hadamard rotation.
	// Input/output are flattened row-major: len = n * outDim.
	FWHTBatch(rot Rotator, input []float32, n int) ([]float32, error)

	// FWHTInverseBatch applies the inverse Hadamard rotation.
	FWHTInverseBatch(rot Rotator, input []float32, n int) ([]float32, error)

	// QuantizeMSEBatch performs batch MSE quantization.
	// Returns packed codes and per-vector norms.
	QuantizeMSEBatch(vectors []float32, n, dim int, rot Rotator, cb Codebook) (codes []byte, norms []float32, err error)

	// DequantizeMSEBatch reconstructs vectors from quantized codes.
	DequantizeMSEBatch(codes []byte, norms []float32, n, dim int, rot Rotator, cb Codebook) ([]float32, error)

	// QJLSketchBatch computes batched QJL 1-bit sketches.
	// Returns packed sign bits and per-vector norms.
	QJLSketchBatch(vectors []float32, n, dim, projDim int, seed uint64) (signs []byte, norms []float32, err error)

	// SearchBruteForce performs brute-force inner-product search.
	// Returns k results per query, sorted by score descending.
	SearchBruteForce(params SearchParams) ([][]SearchResult, error)
}

// Codebook wraps a device-side codebook handle.
type Codebook interface {
	Close()
	BitWidth() int
	Size() int
}

// Rotator wraps a device-side rotator handle.
type Rotator interface {
	Close()
	Dim() int
	OutDim() int
}

// SearchParams configures a brute-force search operation.
type SearchParams struct {
	QueryCodes    []byte
	QueryNorms    []float32
	QuerySigns    []byte    // nil for MSE-only
	QueryResNorms []float32 // nil for MSE-only
	NQueries      int

	DBCodes    []byte
	DBNorms    []float32
	DBSigns    []byte    // nil for MSE-only
	DBResNorms []float32 // nil for MSE-only
	NDB        int

	Dim      int
	BitWidth int
	ProjDim  int // 0 for MSE-only
	CB       Codebook
	K        int
}
