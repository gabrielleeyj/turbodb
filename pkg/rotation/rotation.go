// Package rotation provides rotation matrix management for TurboQuant,
// including randomized Hadamard rotators.
package rotation

// Rotator applies a random rotation to vectors, preserving norms.
// Implementations are immutable and goroutine-safe after construction.
//
// For non-power-of-2 dimensions, Apply may return a larger vector (padded to
// the next power of 2). The output dimension is given by OutDim(). Norms are
// preserved in the full output space.
type Rotator interface {
	// Apply returns Π·x (rotated vector). Does not modify x.
	// Output length is OutDim(), which may be >= Dim().
	// Allocates a fresh buffer; for hot paths use ApplyInto.
	Apply(x []float32) []float32
	// ApplyInto writes Π·x into dst and returns dst. dst must have len OutDim().
	// Does not modify x. dst and x may not alias.
	ApplyInto(dst, x []float32) []float32
	// ApplyTranspose returns Π^T·x (inverse rotation). Does not modify x.
	// Input length must be OutDim(), output length is Dim().
	// Allocates a fresh buffer; for hot paths use ApplyTransposeInto.
	ApplyTranspose(x []float32) []float32
	// ApplyTransposeInto writes Π^T·x into dst and returns dst. dst must have
	// len Dim(). Does not modify x. dst and x may not alias.
	ApplyTransposeInto(dst, x []float32) []float32
	// Seed returns the deterministic seed used to generate this rotator.
	Seed() uint64
	// Dim returns the original input vector dimensionality.
	Dim() int
	// OutDim returns the output dimensionality after rotation (may be padded).
	OutDim() int
	// MarshalBinary serializes the rotator to bytes.
	MarshalBinary() ([]byte, error)
}

// UnmarshalRotator deserializes a rotator from bytes produced by MarshalBinary.
func UnmarshalRotator(data []byte) (Rotator, error) {
	return unmarshalRotator(data)
}
