package quantizer

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/gabrielleeyj/turbodb/pkg/codebook"
	"github.com/gabrielleeyj/turbodb/pkg/rotation"
)

func newTestMSEQuantizer(t *testing.T, dim, bitWidth int) *MSEQuantizer {
	t.Helper()
	rot, err := rotation.NewHadamardRotator(dim, 42)
	if err != nil {
		t.Fatalf("NewHadamardRotator: %v", err)
	}
	cb, err := codebook.Load(dim, bitWidth)
	if err != nil {
		t.Fatalf("codebook.Load: %v", err)
	}
	q, err := NewMSEQuantizer(dim, bitWidth, rot, cb)
	if err != nil {
		t.Fatalf("NewMSEQuantizer: %v", err)
	}
	return q
}

func TestMSEQuantizerRoundTrip(t *testing.T) {
	dims := []int{256, 1536}
	bitWidths := []int{1, 2, 3, 4, 5}

	for _, dim := range dims {
		for _, bw := range bitWidths {
			t.Run(fmtCase(dim, bw), func(t *testing.T) {
				q := newTestMSEQuantizer(t, dim, bw)
				rng := rand.New(rand.NewPCG(123, 456)) //nolint:gosec // deterministic test

				var totalMSE float64
				nTrials := 200
				for range nTrials {
					x := randomUnitVec(rng, dim)
					code, err := q.Quantize(x)
					if err != nil {
						t.Fatalf("Quantize: %v", err)
					}

					xHat, err := q.Dequantize(code)
					if err != nil {
						t.Fatalf("Dequantize: %v", err)
					}

					if len(xHat) != dim {
						t.Fatalf("Dequantize output dim %d, want %d", len(xHat), dim)
					}

					mse := computeMSE(x, xHat)
					totalMSE += mse
				}

				avgMSE := totalMSE / float64(nTrials)
				// MSE should decrease with more bits. Rough bounds:
				// b=1: MSE < 1.0, b=4: MSE < 0.05, b=5: MSE < 0.02
				maxMSE := 2.0 / float64(int(1)<<bw)
				if avgMSE > maxMSE {
					t.Errorf("d=%d b=%d: avg MSE = %f, exceeds threshold %f", dim, bw, avgMSE, maxMSE)
				}
			})
		}
	}
}

func TestBitPackingExact(t *testing.T) {
	dims := []int{256, 1536}
	bitWidths := []int{1, 2, 3, 4, 5, 8}

	for _, dim := range dims {
		for _, bw := range bitWidths {
			q := newTestMSEQuantizer(t, dim, bw)
			x := make([]float32, dim)
			for i := range x {
				x[i] = 1.0 / float32(math.Sqrt(float64(dim)))
			}

			code, err := q.Quantize(x)
			if err != nil {
				t.Fatalf("d=%d b=%d: Quantize: %v", dim, bw, err)
			}

			// Expected size: ceil(bw * outDim / 8)
			outDim := code.Dim
			expectedBytes := (bw*outDim + 7) / 8
			if len(code.Indices) != expectedBytes {
				t.Errorf("d=%d b=%d: packed size %d, want %d (outDim=%d)",
					dim, bw, len(code.Indices), expectedBytes, outDim)
			}
		}
	}
}

func TestMSEQuantizerNormPreservation(t *testing.T) {
	q := newTestMSEQuantizer(t, 1536, 4)
	rng := rand.New(rand.NewPCG(999, 888)) //nolint:gosec // deterministic test

	// Non-unit vector: norm should be stored and restored.
	x := make([]float32, 1536)
	for i := range x {
		x[i] = rng.Float32()*10 - 5
	}

	code, err := q.Quantize(x)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}

	origNorm := vecNorm(x)
	if math.Abs(float64(code.Norm-origNorm)) > 1e-5 {
		t.Errorf("stored norm %f != original %f", code.Norm, origNorm)
	}

	xHat, err := q.Dequantize(code)
	if err != nil {
		t.Fatalf("Dequantize: %v", err)
	}

	reconNorm := vecNorm(xHat)
	relErr := math.Abs(float64(reconNorm-origNorm)) / float64(origNorm)
	// Norm should be approximately preserved (within quantization error).
	if relErr > 0.5 {
		t.Errorf("reconstructed norm %f vs original %f, relErr=%f", reconNorm, origNorm, relErr)
	}
}

func TestMSEQuantizerDimMismatch(t *testing.T) {
	q := newTestMSEQuantizer(t, 256, 4)
	_, err := q.Quantize(make([]float32, 128))
	if err == nil {
		t.Error("expected error for dimension mismatch")
	}
}

func TestMSEQuantizerZeroVector(t *testing.T) {
	q := newTestMSEQuantizer(t, 256, 4)
	x := make([]float32, 256) // all zeros

	// Zero-norm vectors are not meaningful for cosine/IP search and must be rejected.
	if _, err := q.Quantize(x); err == nil {
		t.Error("expected error for zero-norm vector")
	}
}

func TestNewMSEQuantizerValidation(t *testing.T) {
	rot, _ := rotation.NewHadamardRotator(256, 42)
	cb, _ := codebook.Load(256, 4)

	_, err := NewMSEQuantizer(0, 4, rot, cb)
	if err == nil {
		t.Error("expected error for dim=0")
	}

	_, err = NewMSEQuantizer(256, 4, nil, cb)
	if err == nil {
		t.Error("expected error for nil rotator")
	}

	_, err = NewMSEQuantizer(256, 4, rot, nil)
	if err == nil {
		t.Error("expected error for nil codebook")
	}
}

// helpers

func fmtCase(dim, bw int) string {
	return "d" + itoa(dim) + "_b" + itoa(bw)
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	s := ""
	for n > 0 {
		s = string(rune('0'+n%10)) + s
		n /= 10
	}
	return s
}

func randomUnitVec(rng *rand.Rand, d int) []float32 {
	x := make([]float32, d)
	var sum float64
	for i := range x {
		v := float32(rng.NormFloat64())
		x[i] = v
		sum += float64(v * v)
	}
	inv := float32(1.0 / math.Sqrt(sum))
	for i := range x {
		x[i] *= inv
	}
	return x
}

func computeMSE(a, b []float32) float64 {
	var sum float64
	for i := range a {
		d := float64(a[i] - b[i])
		sum += d * d
	}
	return sum / float64(len(a))
}
