package quantizer

import (
	"fmt"
	"math/rand/v2"
)

// QJLSketch implements the 1-bit Quantized Johnson-Lindenstrauss transform
// from Definition 1 of the TurboQuant paper.
// It projects a vector using a seeded Gaussian matrix and takes the sign.
type QJLSketch struct {
	dim  int
	seed uint64
	// projDim is the projection dimensionality (number of sign bits output).
	projDim int
}

// NewQJLSketch creates a QJL sketch for d-dimensional vectors.
// projDim controls the number of random projections (sign bits).
// Higher projDim = lower variance but more storage.
func NewQJLSketch(dim, projDim int, seed uint64) (*QJLSketch, error) {
	if dim < 1 {
		return nil, fmt.Errorf("qjl: dim must be >= 1, got %d", dim)
	}
	if projDim < 1 {
		return nil, fmt.Errorf("qjl: projDim must be >= 1, got %d", projDim)
	}
	return &QJLSketch{dim: dim, projDim: projDim, seed: seed}, nil
}

// Sign computes the 1-bit QJL sketch of x: sign(G·x) where G is a random
// Gaussian matrix seeded deterministically. Returns packed sign bits.
// Also returns the L2 norm of x for the inner-product estimator.
func (q *QJLSketch) Sign(x []float32) (signBits []byte, norm float32) {
	norm = vecNorm(x)

	// Each sign bit is the sign of a random projection.
	signs := make([]int, q.projDim)
	rng := rand.New(rand.NewPCG(q.seed, q.seed^0xcafebabe)) //nolint:gosec // deterministic projection

	for i := range q.projDim {
		// Compute dot product of x with i-th row of Gaussian matrix.
		var dot float64
		for j := range q.dim {
			gij := rng.NormFloat64()
			dot += gij * float64(x[j])
		}
		if dot >= 0 {
			signs[i] = 1
		} else {
			signs[i] = 0
		}
	}

	// Pack sign bits.
	nBytes := (q.projDim + 7) / 8
	signBits = make([]byte, nBytes)
	for i, s := range signs {
		if s == 1 {
			signBits[i/8] |= 1 << (i % 8)
		}
	}

	return signBits, norm
}

// EstimateIP estimates <x, y> given x's QJL sketch and the raw vector y.
// Uses the unbiased estimator from the paper.
func (q *QJLSketch) EstimateIP(signBits []byte, xNorm float32, y []float32) float32 {
	// Compute sign(G·y) and match against signBits.
	rng := rand.New(rand.NewPCG(q.seed, q.seed^0xcafebabe)) //nolint:gosec // same seed for matching

	yNorm := vecNorm(y)
	if xNorm < 1e-30 || yNorm < 1e-30 {
		return 0
	}

	var agree int
	for i := range q.projDim {
		// Same random projection as in Sign.
		var dot float64
		for j := range q.dim {
			gij := rng.NormFloat64()
			dot += gij * float64(y[j])
		}
		ySign := 0
		if dot >= 0 {
			ySign = 1
		}

		xSign := int(signBits[i/8]>>(i%8)) & 1
		if xSign == ySign {
			agree++
		}
	}

	// Unbiased estimator: cos(angle) ≈ 1 - 2*(fraction of disagreements)
	// Then <x,y> ≈ ‖x‖·‖y‖·cos(angle)
	disagree := q.projDim - agree
	cosEst := 1.0 - 2.0*float64(disagree)/float64(q.projDim)
	return xNorm * yNorm * float32(cosEst)
}

// ProdCode extends Code with QJL sketch data for inner-product estimation.
type ProdCode struct {
	// MSECode is the base MSE quantized representation (at bitWidth b-1).
	MSECode Code
	// SignBits is the packed QJL sign vector of the residual.
	SignBits []byte
	// ResidualNorm is ‖x - Dequantize(MSECode)‖₂.
	ResidualNorm float32
	// ProjDim is the number of QJL projection dimensions.
	ProjDim int
}

// ProdQuantizer implements Algorithm 2 from the TurboQuant paper:
// 1. Apply MSEQuantizer with bit-width b-1.
// 2. Compute residual r = x - Dequantize(Quantize(x)).
// 3. Apply QJLSketch to r, store sign vector + ‖r‖₂.
type ProdQuantizer struct {
	mseQ   *MSEQuantizer
	qjl    *QJLSketch
	dim    int
	fullBW int // full bit-width (mse uses fullBW-1, qjl uses 1 bit)
}

// NewProdQuantizer creates a product quantizer for the given dimension
// and total bit-width. The MSE quantizer uses (bitWidth-1) bits, and
// the remaining bit budget goes to the QJL sketch.
func NewProdQuantizer(dim, bitWidth int, mseQ *MSEQuantizer, qjl *QJLSketch) (*ProdQuantizer, error) {
	if dim < 1 {
		return nil, fmt.Errorf("prodquantizer: dim must be >= 1")
	}
	if bitWidth < 2 {
		return nil, fmt.Errorf("prodquantizer: bitWidth must be >= 2 (need at least 1 for MSE + 1 for QJL)")
	}
	if mseQ == nil || qjl == nil {
		return nil, fmt.Errorf("prodquantizer: mseQ and qjl must not be nil")
	}
	return &ProdQuantizer{
		mseQ:   mseQ,
		qjl:    qjl,
		dim:    dim,
		fullBW: bitWidth,
	}, nil
}

// Quantize encodes a vector using Algorithm 2.
func (p *ProdQuantizer) Quantize(x []float32) (ProdCode, error) {
	// Step 1: MSE quantize at (b-1) bits.
	mseCode, err := p.mseQ.Quantize(x)
	if err != nil {
		return ProdCode{}, fmt.Errorf("prodquantizer: mse quantize: %w", err)
	}

	// Step 2: Dequantize to get approximation.
	xHat, err := p.mseQ.Dequantize(mseCode)
	if err != nil {
		return ProdCode{}, fmt.Errorf("prodquantizer: mse dequantize: %w", err)
	}

	// Step 3: Compute residual r = x - xHat.
	residual := make([]float32, p.dim)
	for i := range p.dim {
		residual[i] = x[i] - xHat[i]
	}

	// Step 4: QJL sketch of residual.
	signBits, resNorm := p.qjl.Sign(residual)

	return ProdCode{
		MSECode:      mseCode,
		SignBits:     signBits,
		ResidualNorm: resNorm,
		ProjDim:      p.qjl.projDim,
	}, nil
}

// Dequantize reconstructs from the MSE portion only (QJL is for IP estimation).
func (p *ProdQuantizer) Dequantize(c ProdCode) ([]float32, error) {
	return p.mseQ.Dequantize(c.MSECode)
}

// EstimateInnerProduct estimates <y, x> directly from quantized code,
// without full dequantization. Uses the unbiased estimator from Theorem 2.
func (p *ProdQuantizer) EstimateInnerProduct(y []float32, c ProdCode) (float32, error) {
	// Part 1: <y, xHat> from MSE dequantization.
	xHat, err := p.mseQ.Dequantize(c.MSECode)
	if err != nil {
		return 0, fmt.Errorf("prodquantizer: dequantize for IP: %w", err)
	}

	var mseIP float64
	for i := range y {
		mseIP += float64(y[i]) * float64(xHat[i])
	}

	// Part 2: QJL correction for residual contribution.
	qjlIP := p.qjl.EstimateIP(c.SignBits, c.ResidualNorm, y)

	return float32(mseIP) + qjlIP, nil
}

// Dim returns the expected input dimension.
func (p *ProdQuantizer) Dim() int { return p.dim }

// BitWidth returns the total bit-width.
func (p *ProdQuantizer) BitWidth() int { return p.fullBW }

// EstimateIP is a convenience function that estimates <y, x> from a ProdCode.
// This is the primary hot path for CPU-based search.
func EstimateIP(y []float32, code ProdCode, mseQ *MSEQuantizer, qjl *QJLSketch) (float32, error) {
	xHat, err := mseQ.Dequantize(code.MSECode)
	if err != nil {
		return 0, err
	}

	var ip float64
	minLen := len(y)
	if len(xHat) < minLen {
		minLen = len(xHat)
	}
	for i := range minLen {
		ip += float64(y[i]) * float64(xHat[i])
	}

	qjlCorrection := qjl.EstimateIP(code.SignBits, code.ResidualNorm, y)
	return float32(ip) + qjlCorrection, nil
}
