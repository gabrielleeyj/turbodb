package index

import (
	"container/heap"
	"context"
	"fmt"
	"slices"
	"time"

	"github.com/gabrielleeyj/turbodb/pkg/codebook"
	"github.com/gabrielleeyj/turbodb/pkg/quantizer"
	"github.com/gabrielleeyj/turbodb/pkg/rotation"
)

// SealedSegment is an immutable segment containing quantized vectors.
// It is created by sealing a GrowingSegment — all raw vectors are quantized
// and packed into compact codes. Searches use TurboQuant's inner-product
// estimation on the CPU (GPU path is in internal/cuda).
type SealedSegment struct {
	id       string
	dim      int
	bitWidth int
	count    int

	// Quantized data.
	codes []quantizer.Code
	ids   []string
	norms []float32

	// Quantizer components (needed for search dequantization).
	rotator  rotation.Rotator
	cb       *codebook.Codebook
	mseQ     *quantizer.MSEQuantizer

	createdAt time.Time
	sealedAt  time.Time
}

// SealedSegmentConfig holds the inputs for creating a SealedSegment.
type SealedSegmentConfig struct {
	ID       string
	Dim      int
	BitWidth int
	Rotator  rotation.Rotator
	Codebook *codebook.Codebook
}

// Seal converts a GrowingSegment into an immutable SealedSegment by quantizing
// all vectors. The growing segment should not receive writes during sealing.
func Seal(id string, entries []VectorEntry, cfg SealedSegmentConfig) (*SealedSegment, error) {
	if len(entries) == 0 {
		return nil, fmt.Errorf("seal: no entries to seal")
	}
	if cfg.Rotator == nil {
		return nil, fmt.Errorf("seal: rotator must not be nil")
	}
	if cfg.Codebook == nil {
		return nil, fmt.Errorf("seal: codebook must not be nil")
	}

	mseQ, err := quantizer.NewMSEQuantizer(cfg.Dim, cfg.BitWidth, cfg.Rotator, cfg.Codebook)
	if err != nil {
		return nil, fmt.Errorf("seal: create quantizer: %w", err)
	}

	codes := make([]quantizer.Code, len(entries))
	ids := make([]string, len(entries))
	norms := make([]float32, len(entries))

	// Quantize all vectors.
	rawVectors := make([][]float32, len(entries))
	for i, e := range entries {
		ids[i] = e.ID
		rawVectors[i] = e.Values
	}

	codes, err = quantizer.BatchQuantize(context.TODO(), mseQ, rawVectors)
	if err != nil {
		return nil, fmt.Errorf("seal: batch quantize: %w", err)
	}

	for i := range codes {
		norms[i] = codes[i].Norm
	}

	return &SealedSegment{
		id:        id,
		dim:       cfg.Dim,
		bitWidth:  cfg.BitWidth,
		count:     len(entries),
		codes:     codes,
		ids:       ids,
		norms:     norms,
		rotator:   cfg.Rotator,
		cb:        cfg.Codebook,
		mseQ:      mseQ,
		createdAt: time.Now(),
		sealedAt:  time.Now(),
	}, nil
}

// NewSealedSegmentFromData creates a SealedSegment from pre-quantized data.
// Used when loading from disk.
func NewSealedSegmentFromData(id string, dim, bitWidth int, codes []quantizer.Code,
	ids []string, norms []float32, rot rotation.Rotator, cb *codebook.Codebook,
	createdAt, sealedAt time.Time) (*SealedSegment, error) {

	if len(codes) != len(ids) || len(codes) != len(norms) {
		return nil, fmt.Errorf("sealed segment: mismatched lengths: codes=%d, ids=%d, norms=%d",
			len(codes), len(ids), len(norms))
	}

	mseQ, err := quantizer.NewMSEQuantizer(dim, bitWidth, rot, cb)
	if err != nil {
		return nil, fmt.Errorf("sealed segment: create quantizer: %w", err)
	}

	return &SealedSegment{
		id:        id,
		dim:       dim,
		bitWidth:  bitWidth,
		count:     len(codes),
		codes:     codes,
		ids:       ids,
		norms:     norms,
		rotator:   rot,
		cb:        cb,
		mseQ:      mseQ,
		createdAt: createdAt,
		sealedAt:  sealedAt,
	}, nil
}

// ID returns the segment identifier.
func (s *SealedSegment) ID() string { return s.id }

// Type returns SegmentTypeSealed.
func (s *SealedSegment) Type() SegmentType { return SegmentTypeSealed }

// Count returns the number of quantized vectors.
func (s *SealedSegment) Count() int { return s.count }

// Info returns segment metadata.
func (s *SealedSegment) Info() SegmentInfo {
	return SegmentInfo{
		ID:        s.id,
		Type:      SegmentTypeSealed,
		Count:     s.count,
		Dim:       s.dim,
		BitWidth:  s.bitWidth,
		CreatedAt: s.createdAt,
		SealedAt:  s.sealedAt,
	}
}

// Contains reports whether a vector ID exists in this segment.
func (s *SealedSegment) Contains(id string) bool {
	return slices.Contains(s.ids, id)
}

// Search performs inner-product search over the quantized codes.
// Each code is dequantized and the inner product with the query is computed.
// Results are returned in descending score order.
func (s *SealedSegment) Search(query []float32, topK int, tombstones *TombstoneLog) ([]SearchResult, error) {
	if len(query) != s.dim {
		return nil, fmt.Errorf("sealed segment search: expected dim %d, got %d", s.dim, len(query))
	}
	if topK < 1 {
		return nil, fmt.Errorf("sealed segment search: topK must be >= 1")
	}

	h := &minHeap{}
	heap.Init(h)

	for i := range s.count {
		if tombstones != nil && tombstones.IsDeleted(s.ids[i]) {
			continue
		}

		// Dequantize and compute inner product.
		xHat, err := s.mseQ.Dequantize(s.codes[i])
		if err != nil {
			return nil, fmt.Errorf("sealed segment search: dequantize vector %d: %w", i, err)
		}

		score := innerProduct(query, xHat)

		if h.Len() < topK {
			heap.Push(h, SearchResult{ID: s.ids[i], Score: score})
		} else if score > (*h)[0].Score {
			(*h)[0] = SearchResult{ID: s.ids[i], Score: score}
			heap.Fix(h, 0)
		}
	}

	results := make([]SearchResult, h.Len())
	for i := len(results) - 1; i >= 0; i-- {
		results[i] = heap.Pop(h).(SearchResult)
	}
	return results, nil
}

// Codes returns the quantized codes (used for persistence).
func (s *SealedSegment) Codes() []quantizer.Code { return s.codes }

// IDs returns the vector IDs in storage order.
func (s *SealedSegment) IDs() []string { return s.ids }

// Norms returns the per-vector norms.
func (s *SealedSegment) Norms() []float32 { return s.norms }

// Rotator returns the rotation matrix.
func (s *SealedSegment) Rotator() rotation.Rotator { return s.rotator }

// CodebookRef returns the codebook.
func (s *SealedSegment) CodebookRef() *codebook.Codebook { return s.cb }
