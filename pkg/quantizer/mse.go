package quantizer

import (
	"fmt"
	"math"

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

	return &MSEQuantizer{
		dim:      dim,
		bitWidth: bitWidth,
		cb:       cb,
		rot:      rot,
	}, nil
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
	normalized := normalizeVec(x, norm)

	// Apply rotation (may pad to next power of 2).
	rotated := q.rot.Apply(normalized)
	outDim := len(rotated)

	// Quantize each coordinate.
	indices := make([]int, outDim)
	for i, v := range rotated {
		indices[i] = q.cb.NearestIndex(float64(v))
	}

	// Pack bits.
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

	// Look up centroids to reconstruct rotated vector.
	rotated := make([]float32, c.Dim)
	for i, idx := range indices {
		rotated[i] = float32(q.cb.Centroid(idx))
	}

	// Apply inverse rotation (truncates back to original dim).
	unrotated := q.rot.ApplyTranspose(rotated)

	// Rescale by norm.
	result := make([]float32, len(unrotated))
	for i, v := range unrotated {
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

// normalizeVec returns a new unit vector. If norm is ~0, returns a zero vector.
func normalizeVec(x []float32, norm float32) []float32 {
	result := make([]float32, len(x))
	if norm < 1e-30 {
		return result
	}
	inv := 1.0 / norm
	for i, v := range x {
		result[i] = v * inv
	}
	return result
}
