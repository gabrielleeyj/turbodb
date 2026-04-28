package codebook

import (
	"fmt"
	"math"
)

// Density represents a probability density function over the reals.
type Density interface {
	// PDF returns the probability density at x.
	PDF(x float64) float64
	// Support returns the [lo, hi] interval outside which PDF is ~0.
	Support() (lo, hi float64)
}

// BetaDensity is the density of the inner product between a random unit
// vector and a fixed unit vector in R^d. From the TurboQuant paper:
//
//	f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
//
// Supported on [-1, 1].
type BetaDensity struct {
	dim   int
	coeff float64
	dExp  float64 // (d-3)/2
}

// NewBetaDensity creates a BetaDensity for dimension d.
// Returns an error if d < 2.
func NewBetaDensity(d int) (*BetaDensity, error) {
	if d < 2 {
		return nil, fmt.Errorf("codebook: BetaDensity requires d >= 2, got %d", d)
	}
	// Use log-gamma to avoid overflow for large d.
	lgD2, _ := math.Lgamma(float64(d) / 2.0)
	lgDm1_2, _ := math.Lgamma(float64(d-1) / 2.0)
	coeff := math.Exp(lgD2 - lgDm1_2 - 0.5*math.Log(math.Pi))
	return &BetaDensity{
		dim:   d,
		coeff: coeff,
		dExp:  float64(d-3) / 2.0,
	}, nil
}

// PDF returns the probability density at x for the Beta distribution.
func (b *BetaDensity) PDF(x float64) float64 {
	if x <= -1.0 || x >= 1.0 {
		return 0.0
	}
	return b.coeff * math.Pow(1.0-x*x, b.dExp)
}

// Support returns the [-1, 1] interval for the Beta distribution.
func (b *BetaDensity) Support() (float64, float64) {
	return -1.0, 1.0
}

// GaussianDensity is the high-dimensional Gaussian approximation N(0, 1/d)
// used when d >= 256. The inner product distribution converges to this.
type GaussianDensity struct {
	dim    int
	sigma  float64 // sqrt(1/d)
	coeff  float64 // 1 / (sigma * sqrt(2*pi))
	invVar float64 // d / 2
}

// NewGaussianDensity creates a Gaussian density N(0, 1/d).
// Returns an error if d < 1.
func NewGaussianDensity(d int) (*GaussianDensity, error) {
	if d < 1 {
		return nil, fmt.Errorf("codebook: GaussianDensity requires d >= 1, got %d", d)
	}
	sigma := 1.0 / math.Sqrt(float64(d))
	return &GaussianDensity{
		dim:    d,
		sigma:  sigma,
		coeff:  1.0 / (sigma * math.Sqrt(2.0*math.Pi)),
		invVar: float64(d) / 2.0,
	}, nil
}

// PDF returns the probability density at x for the Gaussian distribution.
func (g *GaussianDensity) PDF(x float64) float64 {
	return g.coeff * math.Exp(-x*x*g.invVar)
}

// Support returns the interval covering > 99.99% of the Gaussian mass.
func (g *GaussianDensity) Support() (float64, float64) {
	// 6 sigma covers > 99.99% of mass.
	bound := 6.0 * g.sigma
	return -bound, bound
}

// DensityForDim returns the appropriate density for dimension d.
// Uses GaussianDensity for d >= 256, BetaDensity otherwise.
func DensityForDim(d int) (Density, error) {
	if d >= 256 {
		return NewGaussianDensity(d)
	}
	return NewBetaDensity(d)
}
