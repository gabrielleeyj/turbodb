// Package codebook implements Lloyd-Max codebook generation
// for the TurboQuant scalar quantizer.
package codebook

import "fmt"

// Codebook holds the sorted centroids produced by Lloyd-Max quantization
// for a given dimensionality and bit-width.
type Codebook struct {
	dim       int
	bitWidth  int
	centroids []float64
}

// NewCodebook creates a Codebook from pre-validated centroids.
// Centroids must be sorted in ascending order.
func NewCodebook(dim, bitWidth int, centroids []float64) (*Codebook, error) {
	expected := 1 << bitWidth
	if len(centroids) != expected {
		return nil, fmt.Errorf("codebook: expected %d centroids for %d-bit, got %d", expected, bitWidth, len(centroids))
	}
	if dim < 1 {
		return nil, fmt.Errorf("codebook: dim must be >= 1, got %d", dim)
	}
	if bitWidth < 1 || bitWidth > 8 {
		return nil, fmt.Errorf("codebook: bitWidth must be 1..8, got %d", bitWidth)
	}
	// Verify sorted order.
	for i := 1; i < len(centroids); i++ {
		if centroids[i] < centroids[i-1] {
			return nil, fmt.Errorf("codebook: centroids must be sorted, index %d (%f) < index %d (%f)", i, centroids[i], i-1, centroids[i-1])
		}
	}
	dst := make([]float64, len(centroids))
	copy(dst, centroids)
	return &Codebook{dim: dim, bitWidth: bitWidth, centroids: dst}, nil
}

// Dim returns the vector dimensionality this codebook was generated for.
func (cb *Codebook) Dim() int { return cb.dim }

// BitWidth returns the quantization bit-width.
func (cb *Codebook) BitWidth() int { return cb.bitWidth }

// Size returns the number of centroids (2^bitWidth).
func (cb *Codebook) Size() int { return len(cb.centroids) }

// Centroid returns the i-th centroid value.
func (cb *Codebook) Centroid(i int) float64 { return cb.centroids[i] }

// Centroids returns a copy of all centroid values.
func (cb *Codebook) Centroids() []float64 {
	dst := make([]float64, len(cb.centroids))
	copy(dst, cb.centroids)
	return dst
}

// NearestIndex returns the index of the centroid closest to x
// using binary search on the sorted centroids.
func (cb *Codebook) NearestIndex(x float64) int {
	n := len(cb.centroids)
	if n == 0 {
		return 0
	}
	// Binary search for the insertion point.
	lo, hi := 0, n
	for lo < hi {
		mid := lo + (hi-lo)/2
		if cb.centroids[mid] < x {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	// lo is the first centroid >= x. Compare lo and lo-1.
	if lo == 0 {
		return 0
	}
	if lo == n {
		return n - 1
	}
	if x-cb.centroids[lo-1] <= cb.centroids[lo]-x {
		return lo - 1
	}
	return lo
}
