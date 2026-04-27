package codebook

import (
	"context"
	"fmt"
	"math"
)

// LloydMaxConfig controls the Lloyd-Max solver.
type LloydMaxConfig struct {
	// Density is the target probability distribution.
	Density Density
	// BitWidth is the number of quantization bits (1..8).
	BitWidth int
	// Tolerance is the convergence threshold for relative distortion change.
	Tolerance float64
	// MaxIter is the maximum number of iterations.
	MaxIter int
	// NumQuadPoints is the number of quadrature points for numerical integration.
	// Higher values give better accuracy at the cost of speed.
	NumQuadPoints int
}

// DefaultLloydMaxConfig returns a config with sensible defaults.
func DefaultLloydMaxConfig(density Density, bitWidth int) LloydMaxConfig {
	return LloydMaxConfig{
		Density:       density,
		BitWidth:      bitWidth,
		Tolerance:     1e-5,
		MaxIter:       500,
		NumQuadPoints: 20000,
	}
}

// LloydMaxResult holds the output of the Lloyd-Max solver.
type LloydMaxResult struct {
	// Centroids are the optimal quantization levels, sorted ascending.
	Centroids []float64
	// Boundaries are the decision boundaries between centroids.
	// len(Boundaries) == len(Centroids) - 1.
	Boundaries []float64
	// Distortion is the final mean squared error.
	Distortion float64
	// Iterations is the number of iterations performed.
	Iterations int
	// Converged indicates whether the solver met the tolerance.
	Converged bool
}

// SolveLloydMax runs the iterative Lloyd-Max algorithm to find optimal
// scalar quantization centroids for the given density.
// The context allows callers to cancel or time out long-running solves.
func SolveLloydMax(ctx context.Context, cfg LloydMaxConfig) (*LloydMaxResult, error) {
	if cfg.Density == nil {
		return nil, fmt.Errorf("lloyd_max: density is nil")
	}
	if cfg.BitWidth < 1 || cfg.BitWidth > 8 {
		return nil, fmt.Errorf("lloyd_max: bitWidth must be 1..8, got %d", cfg.BitWidth)
	}
	if cfg.MaxIter < 1 {
		return nil, fmt.Errorf("lloyd_max: maxIter must be >= 1")
	}
	if cfg.NumQuadPoints < 100 {
		return nil, fmt.Errorf("lloyd_max: numQuadPoints must be >= 100")
	}

	n := 1 << cfg.BitWidth
	lo, hi := cfg.Density.Support()

	// For symmetric distributions (Beta and Gaussian are both symmetric around 0),
	// solve only the positive half and mirror. This halves the problem size and
	// guarantees exact symmetry.
	halfN := n / 2
	posCentroids := initCentroidsQuantile(cfg.Density, 0, hi, halfN, cfg.NumQuadPoints)

	posBoundaries := make([]float64, halfN-1)
	prevPos := make([]float64, halfN)
	converged := false
	iter := 0
	var distortion float64

	for iter = 1; iter <= cfg.MaxIter; iter++ {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("lloyd_max: %w", err)
		}
		copy(prevPos, posCentroids)

		// Compute boundaries as midpoints.
		for i := range posBoundaries {
			posBoundaries[i] = (posCentroids[i] + posCentroids[i+1]) / 2.0
		}

		// Recompute centroids as conditional means over each positive bucket.
		distortion = 0.0
		for i := range posCentroids {
			bucketLo := 0.0
			if i > 0 {
				bucketLo = posBoundaries[i-1]
			}
			bucketHi := hi
			if i < halfN-1 {
				bucketHi = posBoundaries[i]
			}

			num, den := integrateConditionalMean(cfg.Density, bucketLo, bucketHi, cfg.NumQuadPoints)
			if den > 0 {
				newC := num / den
				// Overrelaxation: step 1.5x in the update direction.
				// This accelerates the linear convergence of Lloyd-Max.
				const omega = 1.7
				posCentroids[i] = prevPos[i] + omega*(newC-prevPos[i])
			}

			d := integrateDistortion(cfg.Density, bucketLo, bucketHi, posCentroids[i], cfg.NumQuadPoints)
			distortion += 2 * d // symmetric: negative half has same distortion
		}

		// Convergence: max centroid movement relative to support range.
		maxMove := 0.0
		for i := range posCentroids {
			move := math.Abs(posCentroids[i] - prevPos[i])
			if move > maxMove {
				maxMove = move
			}
		}
		if maxMove/(hi-lo) < cfg.Tolerance {
			converged = true
			break
		}
	}

	// Mirror positive centroids to build the full codebook.
	centroids := make([]float64, n)
	for i := range halfN {
		centroids[i] = -posCentroids[halfN-1-i]
		centroids[halfN+i] = posCentroids[i]
	}

	// Build full boundaries.
	boundaries := make([]float64, n-1)
	for i := range boundaries {
		boundaries[i] = (centroids[i] + centroids[i+1]) / 2.0
	}

	return &LloydMaxResult{
		Centroids:  centroids,
		Boundaries: boundaries,
		Distortion: distortion,
		Iterations: iter,
		Converged:  converged,
	}, nil
}

// initCentroidsQuantile places initial centroids at the conditional means
// of equal-probability quantiles of the density. This gives much faster
// convergence than uniform initialization.
func initCentroidsQuantile(d Density, lo, hi float64, n, nQuad int) []float64 {
	// Build CDF using numerical integration.
	if nQuad%2 != 0 {
		nQuad++
	}
	h := (hi - lo) / float64(nQuad)
	cdf := make([]float64, nQuad+1)
	xs := make([]float64, nQuad+1)
	for i := 0; i <= nQuad; i++ {
		xs[i] = lo + float64(i)*h
	}

	// Trapezoidal rule for CDF.
	cdf[0] = 0
	for i := 1; i <= nQuad; i++ {
		cdf[i] = cdf[i-1] + 0.5*h*(d.PDF(xs[i-1])+d.PDF(xs[i]))
	}

	// Normalize CDF.
	total := cdf[nQuad]
	if total > 0 {
		for i := range cdf {
			cdf[i] /= total
		}
	}

	// Find quantile boundaries that divide into n equal-probability regions.
	boundaries := make([]float64, n+1)
	boundaries[0] = lo
	boundaries[n] = hi
	for i := 1; i < n; i++ {
		target := float64(i) / float64(n)
		// Binary search in CDF.
		idx := 0
		for j := 1; j <= nQuad; j++ {
			if cdf[j] >= target {
				idx = j - 1
				break
			}
		}
		// Linear interpolation.
		if cdf[idx+1] > cdf[idx] {
			frac := (target - cdf[idx]) / (cdf[idx+1] - cdf[idx])
			boundaries[i] = xs[idx] + frac*h
		} else {
			boundaries[i] = xs[idx]
		}
	}

	// Place centroids as conditional means within each quantile bucket.
	centroids := make([]float64, n)
	for i := 0; i < n; i++ {
		num, den := integrateConditionalMean(d, boundaries[i], boundaries[i+1], nQuad/n)
		if den > 0 {
			centroids[i] = num / den
		} else {
			centroids[i] = (boundaries[i] + boundaries[i+1]) / 2.0
		}
	}

	return centroids
}

// integrateConditionalMean computes ∫ x·f(x) dx and ∫ f(x) dx over [a, b]
// using composite Simpson's rule.
func integrateConditionalMean(d Density, a, b float64, n int) (numerator, denominator float64) {
	if a >= b {
		return 0, 0
	}
	// Ensure n is even for Simpson's rule.
	if n%2 != 0 {
		n++
	}
	h := (b - a) / float64(n)

	for i := 0; i <= n; i++ {
		x := a + float64(i)*h
		fx := d.PDF(x)

		var w float64
		switch {
		case i == 0 || i == n:
			w = 1.0
		case i%2 == 1:
			w = 4.0
		default:
			w = 2.0
		}

		denominator += w * fx
		numerator += w * x * fx
	}

	denominator *= h / 3.0
	numerator *= h / 3.0
	return numerator, denominator
}

// integrateDistortion computes ∫ (x - c)^2 · f(x) dx over [a, b].
func integrateDistortion(d Density, a, b, c float64, n int) float64 {
	if a >= b {
		return 0
	}
	if n%2 != 0 {
		n++
	}
	h := (b - a) / float64(n)
	sum := 0.0

	for i := 0; i <= n; i++ {
		x := a + float64(i)*h
		fx := d.PDF(x)
		diff := x - c

		var w float64
		switch {
		case i == 0 || i == n:
			w = 1.0
		case i%2 == 1:
			w = 4.0
		default:
			w = 2.0
		}

		sum += w * diff * diff * fx
	}

	return sum * h / 3.0
}
