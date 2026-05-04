// Package search implements the query planner for TurboDB collections.
//
// The planner sits above pkg/index.Collection and handles oversearch + rerank
// strategies that are independent of how individual segments perform their
// nearest-neighbour search. It exposes the knobs surfaced in the gRPC
// SearchRequest (top_k, rerank, ef_search, exact) so that the engine layer
// can stay thin.
package search

import (
	"fmt"

	"github.com/gabrielleeyj/turbodb/pkg/index"
)

// Options controls a single Search invocation.
type Options struct {
	// TopK is the number of final results to return. Must be 1..MaxTopK.
	TopK int

	// OversearchFactor multiplies the per-segment candidate count. A value
	// of 2.0 means each segment returns 2*TopK results which are then merged
	// down to TopK after optional reranking. Must be >= 1.0; defaults to 1.0
	// when zero.
	OversearchFactor float64

	// Rerank, when true, runs the planner's Reranker over the merged candidate
	// set before truncating to TopK. Without a configured Reranker the flag is
	// a no-op (the planner records this in Plan.Reranked = false).
	Rerank bool

	// Exact, when true, requests a brute-force search path. The MVP planner
	// disables oversearch in this mode and leaves segment selection to the
	// caller (sealed segments still answer queries via dequantized inner
	// product, which is exact for the MSE quantizer).
	Exact bool

	// EfSearch is an advisory candidate-list size for graph-based segments
	// (e.g. CAGRA). It is recorded in Plan.EfSearch but not yet acted on.
	EfSearch int
}

// MaxTopK is the upper bound enforced on Options.TopK.
const MaxTopK = 1000

// DefaultOversearchFactor is applied when Options.OversearchFactor == 0.
const DefaultOversearchFactor = 1.0

// Validate checks the option ranges and returns a normalized copy with
// defaults applied.
func (o Options) Validate() (Options, error) {
	if o.TopK < 1 {
		return o, fmt.Errorf("search: top_k must be >= 1, got %d", o.TopK)
	}
	if o.TopK > MaxTopK {
		return o, fmt.Errorf("search: top_k must be <= %d, got %d", MaxTopK, o.TopK)
	}
	if o.OversearchFactor == 0 {
		o.OversearchFactor = DefaultOversearchFactor
	}
	if o.OversearchFactor < 1.0 {
		return o, fmt.Errorf("search: oversearch_factor must be >= 1.0, got %f", o.OversearchFactor)
	}
	if o.EfSearch < 0 {
		return o, fmt.Errorf("search: ef_search must be >= 0, got %d", o.EfSearch)
	}
	if o.EfSearch != 0 && o.EfSearch < o.TopK {
		return o, fmt.Errorf("search: ef_search must be 0 or >= top_k (%d), got %d", o.TopK, o.EfSearch)
	}
	return o, nil
}

// EffectiveTopK returns the per-segment candidate count after applying the
// oversearch factor. The result is clamped to MaxTopK.
func (o Options) EffectiveTopK() int {
	if o.Exact {
		return o.TopK
	}
	k := int(float64(o.TopK)*o.OversearchFactor + 0.5)
	if k < o.TopK {
		k = o.TopK
	}
	if k > MaxTopK {
		k = MaxTopK
	}
	return k
}

// Plan describes how a single search executed. It is populated by Planner.Run
// and is useful for telemetry, debugging, and adaptive-tuning experiments.
type Plan struct {
	// SegmentsSearched is the number of segments the planner queried.
	SegmentsSearched int
	// CandidatesConsidered is the total number of result rows merged across
	// all segments before truncation to TopK.
	CandidatesConsidered int
	// EffectiveTopK is the per-segment K used after oversearch.
	EffectiveTopK int
	// Reranked indicates whether the configured Reranker was invoked.
	Reranked bool
	// Exact indicates whether the request asked for a brute-force search.
	Exact bool
	// EfSearch echoes the option for downstream telemetry.
	EfSearch int
}

// Reranker rescores a candidate set using a stronger (typically more
// expensive) signal than the segments produced. It may reorder the slice and
// return a (possibly smaller) result set.
type Reranker interface {
	Rerank(query []float32, candidates []index.SearchResult) ([]index.SearchResult, error)
}
