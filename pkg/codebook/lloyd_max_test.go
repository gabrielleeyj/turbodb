package codebook

import (
	"context"
	"math"
	"testing"
)

func TestLloydMaxConvergence(t *testing.T) {
	testCases := []struct {
		dim      int
		bitWidth int
	}{
		{128, 1},
		{128, 2},
		{128, 4},
		{256, 1},
		{256, 4},
		{1536, 1},
		{1536, 4},
	}

	for _, tc := range testCases {
		density, err := DensityForDim(tc.dim)
		if err != nil {
			t.Fatalf("d=%d: DensityForDim: %v", tc.dim, err)
		}
		cfg := DefaultLloydMaxConfig(density, tc.bitWidth)
		result, err := SolveLloydMax(context.Background(), cfg)
		if err != nil {
			t.Fatalf("d=%d b=%d: SolveLloydMax failed: %v", tc.dim, tc.bitWidth, err)
		}

		if !result.Converged {
			t.Errorf("d=%d b=%d: did not converge in %d iterations", tc.dim, tc.bitWidth, result.Iterations)
		}

		if result.Iterations >= 100 {
			t.Errorf("d=%d b=%d: took %d iterations, want < 100", tc.dim, tc.bitWidth, result.Iterations)
		}

		if result.Distortion <= 0 {
			t.Errorf("d=%d b=%d: distortion should be positive, got %e", tc.dim, tc.bitWidth, result.Distortion)
		}
	}
}

func TestLloydMaxDistortionDecreases(t *testing.T) {
	// Verify distortion decreases monotonically by running with limited iterations
	// and checking that more iterations never increase distortion.
	density, err := DensityForDim(128)
	if err != nil {
		t.Fatalf("DensityForDim: %v", err)
	}

	var prevDistortion float64
	for maxIter := 1; maxIter <= 20; maxIter++ {
		cfg := LloydMaxConfig{
			Density:       density,
			BitWidth:      2,
			Tolerance:     0, // never converge early
			MaxIter:       maxIter,
			NumQuadPoints: 10000,
		}
		result, err := SolveLloydMax(context.Background(), cfg)
		if err != nil {
			t.Fatalf("SolveLloydMax failed at iter %d: %v", maxIter, err)
		}

		if maxIter > 1 && result.Distortion > prevDistortion+1e-15 {
			t.Errorf("distortion increased at iter %d: %e -> %e", maxIter, prevDistortion, result.Distortion)
		}
		prevDistortion = result.Distortion
	}
}

func TestCodebookValues_b1_d1536(t *testing.T) {
	// For b=1 and d=1536, centroids should be approximately ±sqrt(2/(pi*d)).
	// This is from the paper's analytical result for 1-bit quantization.
	density, err := DensityForDim(1536)
	if err != nil {
		t.Fatalf("DensityForDim: %v", err)
	}
	cfg := DefaultLloydMaxConfig(density, 1)
	result, err := SolveLloydMax(context.Background(), cfg)
	if err != nil {
		t.Fatalf("SolveLloydMax failed: %v", err)
	}

	if len(result.Centroids) != 2 {
		t.Fatalf("expected 2 centroids for b=1, got %d", len(result.Centroids))
	}

	// For the Gaussian approximation N(0, 1/d):
	// The optimal 1-bit centroids are ±E[|X|] where X ~ N(0, 1/d).
	// E[|X|] = sqrt(2/(pi*d)).
	expected := math.Sqrt(2.0 / (math.Pi * 1536.0))

	// Centroids should be [-expected, +expected].
	if math.Abs(result.Centroids[0]+expected)/expected > 0.01 {
		t.Errorf("centroid[0] = %e, want %e (within 1%%)", result.Centroids[0], -expected)
	}
	if math.Abs(result.Centroids[1]-expected)/expected > 0.01 {
		t.Errorf("centroid[1] = %e, want %e (within 1%%)", result.Centroids[1], expected)
	}
}

func TestLloydMaxCentroidsSorted(t *testing.T) {
	density, err := DensityForDim(512)
	if err != nil {
		t.Fatalf("DensityForDim: %v", err)
	}
	cfg := DefaultLloydMaxConfig(density, 4)
	result, err := SolveLloydMax(context.Background(), cfg)
	if err != nil {
		t.Fatalf("SolveLloydMax failed: %v", err)
	}

	for i := 1; i < len(result.Centroids); i++ {
		if result.Centroids[i] <= result.Centroids[i-1] {
			t.Errorf("centroids not sorted: index %d (%f) <= index %d (%f)",
				i, result.Centroids[i], i-1, result.Centroids[i-1])
		}
	}
}

func TestLloydMaxCentroidsSymmetric(t *testing.T) {
	// For symmetric distributions, centroids should be symmetric around 0.
	density, err := DensityForDim(256)
	if err != nil {
		t.Fatalf("DensityForDim: %v", err)
	}
	cfg := DefaultLloydMaxConfig(density, 4)
	result, err := SolveLloydMax(context.Background(), cfg)
	if err != nil {
		t.Fatalf("SolveLloydMax failed: %v", err)
	}

	n := len(result.Centroids)
	for i := 0; i < n/2; i++ {
		if math.Abs(result.Centroids[i]+result.Centroids[n-1-i]) > 1e-6 {
			t.Errorf("centroids not symmetric: c[%d]=%f, c[%d]=%f",
				i, result.Centroids[i], n-1-i, result.Centroids[n-1-i])
		}
	}
}

func TestLloydMaxInvalidConfig(t *testing.T) {
	density, err := DensityForDim(128)
	if err != nil {
		t.Fatalf("DensityForDim: %v", err)
	}

	_, err = SolveLloydMax(context.Background(), LloydMaxConfig{Density: nil, BitWidth: 4, MaxIter: 100, NumQuadPoints: 1000})
	if err == nil {
		t.Error("expected error for nil density")
	}

	_, err = SolveLloydMax(context.Background(), LloydMaxConfig{Density: density, BitWidth: 0, MaxIter: 100, NumQuadPoints: 1000})
	if err == nil {
		t.Error("expected error for bitWidth=0")
	}

	_, err = SolveLloydMax(context.Background(), LloydMaxConfig{Density: density, BitWidth: 4, MaxIter: 0, NumQuadPoints: 1000})
	if err == nil {
		t.Error("expected error for maxIter=0")
	}
}

// asymmetricDensity is a non-symmetric density used to test that the solver
// rejects inputs that would break its mirroring assumption.
type asymmetricDensity struct{}

func (asymmetricDensity) PDF(x float64) float64 {
	if x < 0 || x > 1 {
		return 0
	}
	// 2x on [0, 1] (mean 2/3) — clearly asymmetric.
	return 2 * x
}
func (asymmetricDensity) Support() (float64, float64) { return 0, 1 }

func TestLloydMaxRejectsAsymmetricDensity(t *testing.T) {
	cfg := DefaultLloydMaxConfig(asymmetricDensity{}, 4)
	if _, err := SolveLloydMax(context.Background(), cfg); err == nil {
		t.Error("expected error for asymmetric density")
	}
}
