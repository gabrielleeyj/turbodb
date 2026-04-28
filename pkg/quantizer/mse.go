package quantizer

import (
	"fmt"
	"math"
	"sync"

	"github.com/gabrielleeyj/turbodb/pkg/codebook"
	"github.com/gabrielleeyj/turbodb/pkg/rotation"
)

// MSEQuantizer implements Algorithm 1 from the TurboQuant paper:
// rotate -> per-coordinate scalar quantization -> bit-pack.
type MSEQuantizer struct {
	dim      int
	bitWidth int
	cb       *codebook.Codebook
	rot      rotation.Rotator
	// floatPool yields padDim-sized float32 slices reused for the normalized
	// input and the rotated output. Avoids two allocations per Quantize call.
	floatPool sync.Pool
	// indexPool yields padDim-sized int slices reused for centroid indices.
	indexPool sync.Pool
}

// NewMSEQuantizer creates an MSE quantizer for the given dimension and bit-width.
func NewMSEQuantizer(dim, bitWidth int, rot rotation.Rotator, cb *codebook.Codebook) (*MSEQuantizer, error) {
	if dim < 1 {
		return nil, fmt.Errorf("quantizer: dim must be >= 1, got %d", dim)
	}
	if bitWidth < 1 || bitWidth > 8 {
		return nil, fmt.Errorf("quantizer: bitWidth must be 1..8, got %d", bitWidth)
	}
	if rot == nil {
		return nil, fmt.Errorf("quantizer: rotator is nil")
	}
	if cb == nil {
		return nil, fmt.Errorf("quantizer: codebook is nil")
	}
	if rot.Dim() != dim {
		return nil, fmt.Errorf("quantizer: rotator dim %d != quantizer dim %d", rot.Dim(), dim)
	}
	if cb.BitWidth() != bitWidth {
		return nil, fmt.Errorf("quantizer: codebook bitWidth %d != quantizer bitWidth %d", cb.BitWidth(), bitWidth)
	}

	outDim := rot.OutDim()
	q := &MSEQuantizer{
		dim:      dim,
		bitWidth: bitWidth,
		cb:       cb,
		rot:      rot,
	}
	q.floatPool.New = func() any {
		s := make([]float32, outDim)
		return &s
	}
	q.indexPool.New = func() any {
		s := make([]int, outDim)
		return &s
	}
	return q, nil
}

// Quantize encodes a vector following Algorithm 1:
// 1. Compute and store norm, normalize to unit vector.
// 2. Apply rotation.
// 3. For each coordinate, find nearest codebook centroid.
// 4. Pack indices into bit-stream.
func (q *MSEQuantizer) Quantize(x []float32) (Code, error) {
	if len(x) != q.dim {
		return Code{}, fmt.Errorf("quantizer: input dim %d != expected %d", len(x), q.dim)
	}
	if containsNaNOrInf(x) {
		return Code{}, fmt.Errorf("quantizer: input contains NaN or Inf")
	}

	// Compute norm and normalize.
	norm := vecNorm(x)
	if norm < 1e-30 {
		return Code{}, fmt.Errorf("quantizer: input vector has zero norm")
	}

	// Borrow a float32 buffer from the pool and write x/norm into the first
	// dim slots, zeroing the tail. This avoids the per-call allocation that
	// normalizeVec would otherwise make.
	outDim := q.rot.OutDim()
	normPtr := q.floatPool.Get().(*[]float32)
	defer q.floatPool.Put(normPtr)
	normalized := (*normPtr)[:outDim]
	inv := 1.0 / norm
	for i, v := range x {
		normalized[i] = v * inv
	}
	for i := q.dim; i < outDim; i++ {
		normalized[i] = 0
	}

	// Apply rotation into another pooled buffer (no alloc).
	rotPtr := q.floatPool.Get().(*[]float32)
	defer q.floatPool.Put(rotPtr)
	rotated := q.rot.ApplyInto((*rotPtr)[:outDim], normalized)

	// Quantize each coordinate using a pooled index slice.
	idxPtr := q.indexPool.Get().(*[]int)
	defer q.indexPool.Put(idxPtr)
	indices := (*idxPtr)[:outDim]
	for i, v := range rotated {
		indices[i] = q.cb.NearestIndex(float64(v))
	}

	// Pack bits — the returned slice escapes into Code so it is allocated.
	packed, err := PackBits(indices, q.bitWidth)
	if err != nil {
		return Code{}, fmt.Errorf("quantizer: pack failed: %w", err)
	}

	return Code{
		Indices:  packed,
		Norm:     norm,
		BitWidth: q.bitWidth,
		Dim:      outDim,
	}, nil
}

// Dequantize reconstructs an approximate vector from a Code:
// 1. Unpack indices.
// 2. Look up centroids.
// 3. Apply inverse rotation.
// 4. Rescale by stored norm.
func (q *MSEQuantizer) Dequantize(c Code) ([]float32, error) {
	if c.BitWidth != q.bitWidth {
		return nil, fmt.Errorf("quantizer: code bitWidth %d != quantizer %d", c.BitWidth, q.bitWidth)
	}

	// Unpack indices.
	indices, err := UnpackBits(c.Indices, c.BitWidth, c.Dim)
	if err != nil {
		return nil, fmt.Errorf("quantizer: unpack failed: %w", err)
	}

	// Look up centroids into a pooled buffer (no alloc).
	rotPtr := q.floatPool.Get().(*[]float32)
	defer q.floatPool.Put(rotPtr)
	rotated := (*rotPtr)[:c.Dim]
	for i, idx := range indices {
		rotated[i] = float32(q.cb.Centroid(idx))
	}

	// Apply inverse rotation directly into the result buffer.
	result := make([]float32, q.dim)
	q.rot.ApplyTransposeInto(result, rotated)

	// Rescale by norm in place.
	for i, v := range result {
		result[i] = v * c.Norm
	}

	return result, nil
}

// BitWidth returns the quantization bit-width.
func (q *MSEQuantizer) BitWidth() int { return q.bitWidth }

// Dim returns the expected input vector dimensionality.
func (q *MSEQuantizer) Dim() int { return q.dim }

// Codebook returns the codebook used for quantization.
func (q *MSEQuantizer) Codebook() *codebook.Codebook { return q.cb }

// Rotator returns the rotation used before quantization.
func (q *MSEQuantizer) Rotator() rotation.Rotator { return q.rot }

// vecNorm computes the L2 norm of a float32 vector.
func vecNorm(x []float32) float32 {
	var sum float64
	for _, v := range x {
		sum += float64(v) * float64(v)
	}
	return float32(math.Sqrt(sum))
}

// containsNaNOrInf returns true if any element is NaN or ±Inf.
func containsNaNOrInf(x []float32) bool {
	for _, v := range x {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return true
		}
	}
	return false
}

