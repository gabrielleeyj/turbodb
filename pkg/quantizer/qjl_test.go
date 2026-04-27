package quantizer

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/gabrielleeyj/turbodb/pkg/codebook"
	"github.com/gabrielleeyj/turbodb/pkg/rotation"
)

func newTestProdQuantizer(t *testing.T, dim, bitWidth int) (*ProdQuantizer, *MSEQuantizer, *QJLSketch) {
	t.Helper()
	rot, err := rotation.NewHadamardRotator(dim, 42)
	if err != nil {
		t.Fatalf("NewHadamardRotator: %v", err)
	}
	// MSE quantizer uses bitWidth-1.
	mseBW := bitWidth - 1
	cb, err := codebook.Load(dim, mseBW)
	if err != nil {
		t.Fatalf("codebook.Load: %v", err)
	}
	mseQ, err := NewMSEQuantizer(dim, mseBW, rot, cb)
	if err != nil {
		t.Fatalf("NewMSEQuantizer: %v", err)
	}

	// QJL with dim projections (matching the paper).
	qjl, err := NewQJLSketch(dim, dim, 99)
	if err != nil {
		t.Fatalf("NewQJLSketch: %v", err)
	}

	pq, err := NewProdQuantizer(dim, bitWidth, mseQ, qjl)
	if err != nil {
		t.Fatalf("NewProdQuantizer: %v", err)
	}

	return pq, mseQ, qjl
}

func TestProdQuantizerUnbiased(t *testing.T) {
	dim := 256
	bw := 4
	pq, _, _ := newTestProdQuantizer(t, dim, bw)
	rng := rand.New(rand.NewPCG(123, 456)) //nolint:gosec // deterministic test

	nSamples := 500
	var totalError float64

	for range nSamples {
		x := randomUnitVec(rng, dim)
		y := randomUnitVec(rng, dim)

		trueIP := innerProduct(x, y)

		code, err := pq.Quantize(x)
		if err != nil {
			t.Fatalf("Quantize: %v", err)
		}

		estIP, err := pq.EstimateInnerProduct(y, code)
		if err != nil {
			t.Fatalf("EstimateInnerProduct: %v", err)
		}

		totalError += float64(estIP) - trueIP
	}

	meanError := totalError / float64(nSamples)
	// Welch's t-test: mean should be statistically indistinguishable from 0.
	// For a rough check, mean error should be small relative to expected IP magnitude.
	if math.Abs(meanError) > 0.1 {
		t.Errorf("mean estimation error = %f, should be ~0 (unbiased)", meanError)
	}
}

func TestProdQuantizerVariance(t *testing.T) {
	dim := 256
	bw := 4
	pq, _, _ := newTestProdQuantizer(t, dim, bw)
	rng := rand.New(rand.NewPCG(789, 101)) //nolint:gosec // deterministic test

	nSamples := 500
	var totalSqError float64

	for range nSamples {
		x := randomUnitVec(rng, dim)
		y := randomUnitVec(rng, dim)

		trueIP := innerProduct(x, y)

		code, err := pq.Quantize(x)
		if err != nil {
			t.Fatalf("Quantize: %v", err)
		}

		estIP, err := pq.EstimateInnerProduct(y, code)
		if err != nil {
			t.Fatalf("EstimateInnerProduct: %v", err)
		}

		diff := float64(estIP) - trueIP
		totalSqError += diff * diff
	}

	variance := totalSqError / float64(nSamples)
	// Variance should be finite and decrease with more bits.
	// For b=4 (mse uses b=3), variance should be relatively small.
	if variance > 1.0 {
		t.Errorf("estimation variance = %f, seems too high", variance)
	}
	t.Logf("d=%d b=%d: estimation variance = %e", dim, bw, variance)
}

func TestProdQuantizerDequantize(t *testing.T) {
	dim := 128
	bw := 3
	pq, _, _ := newTestProdQuantizer(t, dim, bw)
	rng := rand.New(rand.NewPCG(555, 666)) //nolint:gosec // deterministic test

	x := randomUnitVec(rng, dim)
	code, err := pq.Quantize(x)
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}

	xHat, err := pq.Dequantize(code)
	if err != nil {
		t.Fatalf("Dequantize: %v", err)
	}

	if len(xHat) != dim {
		t.Errorf("Dequantize output dim %d, want %d", len(xHat), dim)
	}

	// Should be a reasonable approximation.
	mse := computeMSE(x, xHat)
	if mse > 1.0 {
		t.Errorf("MSE = %f, seems too high", mse)
	}
}

func TestQJLSketchDeterminism(t *testing.T) {
	qjl, _ := NewQJLSketch(128, 128, 42)
	x := make([]float32, 128)
	for i := range x {
		x[i] = float32(i) / 128.0
	}

	sign1, norm1, err := qjl.Sign(x)
	if err != nil {
		t.Fatalf("Sign: %v", err)
	}
	sign2, norm2, err := qjl.Sign(x)
	if err != nil {
		t.Fatalf("Sign: %v", err)
	}

	if norm1 != norm2 {
		t.Errorf("norm mismatch: %f != %f", norm1, norm2)
	}
	for i := range sign1 {
		if sign1[i] != sign2[i] {
			t.Fatalf("sign bit mismatch at byte %d", i)
		}
	}
}

func innerProduct(a, b []float32) float64 {
	var sum float64
	for i := range a {
		if i < len(b) {
			sum += float64(a[i]) * float64(b[i])
		}
	}
	return sum
}

func TestNewProdQuantizerValidation(t *testing.T) {
	_, err := NewProdQuantizer(0, 4, nil, nil)
	if err == nil {
		t.Error("expected error for dim=0")
	}

	_, err = NewProdQuantizer(128, 1, nil, nil)
	if err == nil {
		t.Error("expected error for bitWidth=1")
	}
}
