package search

import (
	"container/heap"
	"context"
	"fmt"

	"github.com/gabrielleeyj/turbodb/pkg/index"
)

// Planner orchestrates a single Search request against a Collection.
type Planner struct {
	coll     *index.Collection
	reranker Reranker
}

// NewPlanner returns a Planner bound to the given collection. The reranker may
// be nil; when nil, Options.Rerank has no effect.
func NewPlanner(coll *index.Collection, reranker Reranker) (*Planner, error) {
	if coll == nil {
		return nil, fmt.Errorf("search: collection must not be nil")
	}
	return &Planner{coll: coll, reranker: reranker}, nil
}

// Run executes a search according to opts and returns the top results plus a
// description of the path taken.
func (p *Planner) Run(ctx context.Context, query []float32, opts Options) ([]index.SearchResult, Plan, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, Plan{}, err
	}

	normalized, err := opts.Validate()
	if err != nil {
		return nil, Plan{}, err
	}
	if len(query) == 0 {
		return nil, Plan{}, fmt.Errorf("search: query must not be empty")
	}
	if len(query) != p.coll.Dim() {
		return nil, Plan{}, fmt.Errorf("search: expected dim %d, got %d", p.coll.Dim(), len(query))
	}

	stats := p.coll.Stats()
	plan := Plan{
		SegmentsSearched: stats.GrowingSegmentCount + stats.SealedSegmentCount,
		EffectiveTopK:    normalized.EffectiveTopK(),
		Exact:            normalized.Exact,
		EfSearch:         normalized.EfSearch,
	}

	// Per-segment candidate fan-out lives inside the collection. Calling with
	// EffectiveTopK gives us oversearched candidates we can then merge/rerank.
	candidates, err := p.coll.Search(query, plan.EffectiveTopK)
	if err != nil {
		return nil, plan, fmt.Errorf("search: collection: %w", err)
	}
	plan.CandidatesConsidered = len(candidates)

	if normalized.Rerank && p.reranker != nil {
		reranked, err := p.reranker.Rerank(query, candidates)
		if err != nil {
			return nil, plan, fmt.Errorf("search: rerank: %w", err)
		}
		candidates = reranked
		plan.Reranked = true
	}

	return takeTopK(candidates, normalized.TopK), plan, nil
}

// takeTopK returns the top-K results from candidates by descending score.
// When len(candidates) <= K, the slice is sorted in place and returned.
func takeTopK(candidates []index.SearchResult, k int) []index.SearchResult {
	if k <= 0 {
		return nil
	}
	if len(candidates) <= k {
		sortByScoreDesc(candidates)
		return candidates
	}

	// Bounded min-heap to find top-K by descending score.
	h := &resultHeap{}
	heap.Init(h)
	for _, r := range candidates {
		if h.Len() < k {
			heap.Push(h, r)
		} else if r.Score > (*h)[0].Score {
			(*h)[0] = r
			heap.Fix(h, 0)
		}
	}

	out := make([]index.SearchResult, h.Len())
	for i := len(out) - 1; i >= 0; i-- {
		out[i] = heap.Pop(h).(index.SearchResult)
	}
	return out
}

func sortByScoreDesc(rs []index.SearchResult) {
	// Insertion sort: K is small and rs is already partially ordered after
	// per-segment merges.
	for i := 1; i < len(rs); i++ {
		x := rs[i]
		j := i - 1
		for j >= 0 && rs[j].Score < x.Score {
			rs[j+1] = rs[j]
			j--
		}
		rs[j+1] = x
	}
}

// resultHeap is a min-heap of SearchResult ordered by ascending score, used to
// keep the top-K elements during merge.
type resultHeap []index.SearchResult

func (h resultHeap) Len() int           { return len(h) }
func (h resultHeap) Less(i, j int) bool { return h[i].Score < h[j].Score }
func (h resultHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *resultHeap) Push(x any)        { *h = append(*h, x.(index.SearchResult)) }
func (h *resultHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}
