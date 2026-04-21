// Package quantizer defines the core TurboQuant quantization interfaces
// and provides a pure-Go CPU reference implementation.
package quantizer

import (
	"github.com/gabrielleeyj/turbodb/pkg/codebook"
	"github.com/gabrielleeyj/turbodb/pkg/rotation"
)

// Code is a quantized representation of a vector.
type Code struct {
	// Indices holds the packed bit-stream of codebook indices.
	Indices []byte
	// Norm stores the original vector's L2 norm (for non-unit vectors).
	Norm float32
	// BitWidth is the number of bits per coordinate.
	BitWidth int
	// Dim is the quantized dimensionality (may be padded by rotation).
	Dim int
}

// Quantizer defines the interface for vector quantization.
type Quantizer interface {
	// Quantize encodes a vector into a compact Code.
	Quantize(x []float32) (Code, error)
	// Dequantize reconstructs an approximate vector from a Code.
	Dequantize(c Code) ([]float32, error)
	// BitWidth returns the quantization bit-width.
	BitWidth() int
	// Dim returns the expected input vector dimensionality.
	Dim() int
	// Codebook returns the codebook used for quantization.
	Codebook() *codebook.Codebook
	// Rotator returns the rotation used before quantization.
	Rotator() rotation.Rotator
}
