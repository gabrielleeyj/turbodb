package codebook

import (
	"math"
	"testing"
)

func TestBetaDensityIntegratesToOne(t *testing.T) {
	// Skip d=2: exponent (d-3)/2 = -0.5 creates boundary singularities
	// that Simpson's rule handles poorly. d=2 is not used in practice.
	dims := []int{3, 10, 50, 128}
	for _, d := range dims {
		bd, err := NewBetaDensity(d)
		if err != nil {
			t.Fatalf("NewBetaDensity(%d): %v", d, err)
		}
		lo, hi := bd.Support()
		integral := simpsonIntegrate(bd.PDF, lo, hi, 100000)
		if math.Abs(integral-1.0) > 1e-4 {
			t.Errorf("d=%d: beta density integrates to %f, want 1.0", d, integral)
		}
	}
}

func TestBetaDensitySymmetric(t *testing.T) {
	bd, err := NewBetaDensity(128)
	if err != nil {
		t.Fatalf("NewBetaDensity: %v", err)
	}
	for _, x := range []float64{0.01, 0.05, 0.1, 0.3, 0.5} {
		if math.Abs(bd.PDF(x)-bd.PDF(-x)) > 1e-12 {
			t.Errorf("beta density not symmetric: f(%f)=%f, f(%f)=%f", x, bd.PDF(x), -x, bd.PDF(-x))
		}
	}
}

func TestBetaDensityZeroOutsideSupport(t *testing.T) {
	bd, err := NewBetaDensity(10)
	if err != nil {
		t.Fatalf("NewBetaDensity: %v", err)
	}
	if bd.PDF(-1.0) != 0.0 || bd.PDF(1.0) != 0.0 || bd.PDF(-2.0) != 0.0 || bd.PDF(2.0) != 0.0 {
		t.Error("beta density should be zero at and outside boundaries")
	}
}

func TestBetaDensityInvalidDim(t *testing.T) {
	_, err := NewBetaDensity(1)
	if err == nil {
		t.Error("expected error for d=1")
	}
	_, err = NewBetaDensity(0)
	if err == nil {
		t.Error("expected error for d=0")
	}
}

func TestGaussianDensityIntegratesToOne(t *testing.T) {
	dims := []int{256, 512, 1536, 4096}
	for _, d := range dims {
		gd, err := NewGaussianDensity(d)
		if err != nil {
			t.Fatalf("NewGaussianDensity(%d): %v", d, err)
		}
		lo, hi := gd.Support()
		integral := simpsonIntegrate(gd.PDF, lo, hi, 100000)
		if math.Abs(integral-1.0) > 1e-4 {
			t.Errorf("d=%d: gaussian density integrates to %f, want 1.0", d, integral)
		}
	}
}

func TestGaussianDensityVariance(t *testing.T) {
	d := 1536
	gd, err := NewGaussianDensity(d)
	if err != nil {
		t.Fatalf("NewGaussianDensity: %v", err)
	}
	lo, hi := gd.Support()

	// Variance should be 1/d.
	variance := simpsonIntegrate(func(x float64) float64 {
		return x * x * gd.PDF(x)
	}, lo, hi, 100000)

	expected := 1.0 / float64(d)
	if math.Abs(variance-expected)/expected > 0.01 {
		t.Errorf("d=%d: gaussian variance = %e, want %e", d, variance, expected)
	}
}

func TestGaussianDensityInvalidDim(t *testing.T) {
	_, err := NewGaussianDensity(0)
	if err == nil {
		t.Error("expected error for d=0")
	}
	_, err = NewGaussianDensity(-1)
	if err == nil {
		t.Error("expected error for d=-1")
	}
}

func TestDensityForDimSelection(t *testing.T) {
	// d < 256 should return BetaDensity.
	d1, err := DensityForDim(128)
	if err != nil {
		t.Fatalf("DensityForDim(128): %v", err)
	}
	if _, ok := d1.(*BetaDensity); !ok {
		t.Error("DensityForDim(128) should return *BetaDensity")
	}
	// d >= 256 should return GaussianDensity.
	d2, err := DensityForDim(256)
	if err != nil {
		t.Fatalf("DensityForDim(256): %v", err)
	}
	if _, ok := d2.(*GaussianDensity); !ok {
		t.Error("DensityForDim(256) should return *GaussianDensity")
	}
}

// simpsonIntegrate computes ∫ f(x) dx over [a, b] using composite Simpson's rule.
func simpsonIntegrate(f func(float64) float64, a, b float64, n int) float64 {
	if n%2 != 0 {
		n++
	}
	h := (b - a) / float64(n)
	sum := f(a) + f(b)
	for i := 1; i < n; i++ {
		x := a + float64(i)*h
		if i%2 == 0 {
			sum += 2.0 * f(x)
		} else {
			sum += 4.0 * f(x)
		}
	}
	return sum * h / 3.0
}
