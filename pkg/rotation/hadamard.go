package rotation

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
)

const hadamardTypeTag = 1

// HadamardRotator implements randomized Fast Walsh-Hadamard Transform rotation.
// It applies: sign_flip -> FWHT -> sign_flip, which is O(d log d).
// The input dimension is padded to the next power of 2 internally.
type HadamardRotator struct {
	dim     int     // original input dimension
	padDim  int     // next power of 2 >= dim
	seed    uint64  // deterministic seed
	signs1  []int8  // first random sign vector (+1/-1), length padDim
	signs2  []int8  // second random sign vector (+1/-1), length padDim
	normInv float32 // 1/sqrt(padDim) normalization factor
}

// NewHadamardRotator creates a randomized Hadamard rotator for dimension d,
// deterministic from the given seed. Thread-safe after construction.
func NewHadamardRotator(d int, seed uint64) (*HadamardRotator, error) {
	if d < 1 {
		return nil, fmt.Errorf("rotation: dim must be >= 1, got %d", d)
	}

	padDim := nextPow2(d)
	rng := rand.New(rand.NewPCG(seed, seed^0xdeadbeef)) //nolint:gosec // deterministic seed is intentional

	signs1 := make([]int8, padDim)
	signs2 := make([]int8, padDim)
	for i := range padDim {
		if rng.IntN(2) == 0 {
			signs1[i] = 1
		} else {
			signs1[i] = -1
		}
		if rng.IntN(2) == 0 {
			signs2[i] = 1
		} else {
			signs2[i] = -1
		}
	}

	return &HadamardRotator{
		dim:     d,
		padDim:  padDim,
		seed:    seed,
		signs1:  signs1,
		signs2:  signs2,
		normInv: 1.0 / float32(math.Sqrt(float64(padDim))),
	}, nil
}

// Apply returns Π·x. Does not modify x.
// Output length is padDim (next power of 2 >= dim).
func (h *HadamardRotator) Apply(x []float32) []float32 {
	buf := h.pad(x)

	// Step 1: multiply by first random sign vector.
	for i := range buf {
		buf[i] *= float32(h.signs1[i])
	}

	// Step 2: FWHT.
	fwht(buf)

	// Step 3: normalize.
	for i := range buf {
		buf[i] *= h.normInv
	}

	// Step 4: multiply by second random sign vector.
	for i := range buf {
		buf[i] *= float32(h.signs2[i])
	}

	return buf
}

// ApplyTranspose returns Π^T·x. Does not modify x.
// Input length must be padDim (OutDim). Output length is dim (Dim).
// For the randomized Hadamard, Π^T = Π^{-1} because Π is orthogonal.
func (h *HadamardRotator) ApplyTranspose(x []float32) []float32 {
	buf := make([]float32, h.padDim)
	copy(buf, x)

	// Step 1: undo second sign flip (self-inverse).
	for i := range buf {
		buf[i] *= float32(h.signs2[i])
	}

	// Step 2: FWHT (self-inverse up to normalization).
	fwht(buf)

	// Step 3: normalize.
	for i := range buf {
		buf[i] *= h.normInv
	}

	// Step 4: undo first sign flip.
	for i := range buf {
		buf[i] *= float32(h.signs1[i])
	}

	return buf[:h.dim]
}

// Seed returns the deterministic seed.
func (h *HadamardRotator) Seed() uint64 { return h.seed }

// Dim returns the input vector dimensionality.
func (h *HadamardRotator) Dim() int { return h.dim }

// OutDim returns the output dimensionality (padded to next power of 2).
func (h *HadamardRotator) OutDim() int { return h.padDim }

// MarshalBinary serializes the rotator.
func (h *HadamardRotator) MarshalBinary() ([]byte, error) {
	// Format: [type_tag:1][dim:4][seed:8] = 13 bytes
	buf := make([]byte, 13)
	buf[0] = hadamardTypeTag
	binary.LittleEndian.PutUint32(buf[1:5], uint32(h.dim)) //nolint:gosec // dim is validated positive in constructor
	binary.LittleEndian.PutUint64(buf[5:13], h.seed)
	return buf, nil
}

// pad copies x into a zero-padded buffer of length padDim.
func (h *HadamardRotator) pad(x []float32) []float32 {
	buf := make([]float32, h.padDim)
	copy(buf, x)
	return buf
}

// fwht performs an in-place Fast Walsh-Hadamard Transform (unnormalized).
func fwht(a []float32) {
	n := len(a)
	for h := 1; h < n; h <<= 1 {
		for i := 0; i < n; i += h << 1 {
			for j := i; j < i+h; j++ {
				x := a[j]
				y := a[j+h]
				a[j] = x + y
				a[j+h] = x - y
			}
		}
	}
}

// nextPow2 returns the smallest power of 2 >= n.
func nextPow2(n int) int {
	if n <= 1 {
		return 1
	}
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

// unmarshalRotator deserializes a rotator from binary data.
func unmarshalRotator(data []byte) (Rotator, error) {
	if len(data) < 1 {
		return nil, fmt.Errorf("rotation: empty data")
	}
	switch data[0] {
	case hadamardTypeTag:
		if len(data) < 13 {
			return nil, fmt.Errorf("rotation: hadamard data too short: %d bytes", len(data))
		}
		dim := int(binary.LittleEndian.Uint32(data[1:5]))
		seed := binary.LittleEndian.Uint64(data[5:13])
		return NewHadamardRotator(dim, seed)
	default:
		return nil, fmt.Errorf("rotation: unknown type tag %d", data[0])
	}
}
