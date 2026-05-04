package search

import (
	"context"
	"errors"
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/gabrielleeyj/turbodb/pkg/codebook"
	"github.com/gabrielleeyj/turbodb/pkg/index"
	"github.com/gabrielleeyj/turbodb/pkg/rotation"
)

const (
	testDim      = 128
	testBitWidth = 4
)

func newCollection(t *testing.T) *index.Collection {
	t.Helper()
	rot, err := rotation.NewHadamardRotator(testDim, 7)
	if err != nil {
		t.Fatal(err)
	}
	cb, err := codebook.Load(testDim, testBitWidth)
	if err != nil {
		t.Fatal(err)
	}
	c, err := index.NewCollection(index.CollectionConfig{
		Name:     "t",
		Dim:      testDim,
		BitWidth: testBitWidth,
		Rotator:  rot,
		Codebook: cb,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = c.Close() })
	return c
}

func seedVectors(t *testing.T, c *index.Collection, n int) [][]float32 {
	t.Helper()
	rng := rand.New(rand.NewPCG(13, 17))
	vecs := make([][]float32, n)
	for i := range n {
		v := make([]float32, testDim)
		for j := range v {
			v[j] = rng.Float32()*2 - 1
		}
		vecs[i] = v
		if err := c.Insert(index.VectorEntry{ID: fmt.Sprintf("id-%04d", i), Values: v}); err != nil {
			t.Fatal(err)
		}
	}
	return vecs
}

func TestOptionsValidate(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name    string
		in      Options
		wantErr bool
	}{
		{"defaults topK only", Options{TopK: 10}, false},
		{"missing topK", Options{}, true},
		{"topK too large", Options{TopK: 5000}, true},
		{"oversearch < 1", Options{TopK: 5, OversearchFactor: 0.5}, true},
		{"ef_search below topK", Options{TopK: 10, EfSearch: 5}, true},
		{"ef_search ok", Options{TopK: 10, EfSearch: 32}, false},
		{"negative ef_search", Options{TopK: 10, EfSearch: -1}, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := tc.in.Validate()
			if tc.wantErr && err == nil {
				t.Fatalf("expected error, got nil")
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestEffectiveTopK(t *testing.T) {
	t.Parallel()
	tests := []struct {
		opts Options
		want int
	}{
		{Options{TopK: 10}, 10},
		{Options{TopK: 10, OversearchFactor: 1.0}, 10},
		{Options{TopK: 10, OversearchFactor: 2.5}, 25},
		{Options{TopK: 10, Exact: true, OversearchFactor: 5.0}, 10},
		{Options{TopK: 600, OversearchFactor: 3.0}, MaxTopK},
	}
	for i, tc := range tests {
		opts, err := tc.opts.Validate()
		if err != nil {
			t.Fatalf("[%d] validate: %v", i, err)
		}
		if got := opts.EffectiveTopK(); got != tc.want {
			t.Errorf("[%d] got %d, want %d", i, got, tc.want)
		}
	}
}

func TestPlannerRunBasic(t *testing.T) {
	t.Parallel()
	coll := newCollection(t)
	vecs := seedVectors(t, coll, 50)

	p, err := NewPlanner(coll, nil)
	if err != nil {
		t.Fatal(err)
	}

	res, plan, err := p.Run(context.Background(), vecs[0], Options{TopK: 5})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 5 {
		t.Errorf("results len = %d, want 5", len(res))
	}
	if res[0].ID != "id-0000" {
		t.Errorf("top id = %q, want id-0000", res[0].ID)
	}
	if plan.SegmentsSearched < 1 {
		t.Errorf("plan.SegmentsSearched = %d, want >= 1", plan.SegmentsSearched)
	}
	if plan.EffectiveTopK != 5 {
		t.Errorf("plan.EffectiveTopK = %d, want 5", plan.EffectiveTopK)
	}
	if plan.Reranked {
		t.Errorf("plan.Reranked = true with no reranker configured")
	}
}

func TestPlannerOversearch(t *testing.T) {
	t.Parallel()
	coll := newCollection(t)
	vecs := seedVectors(t, coll, 60)

	p, _ := NewPlanner(coll, nil)
	_, plan, err := p.Run(context.Background(), vecs[0], Options{TopK: 5, OversearchFactor: 3.0})
	if err != nil {
		t.Fatal(err)
	}
	if plan.EffectiveTopK != 15 {
		t.Errorf("EffectiveTopK = %d, want 15", plan.EffectiveTopK)
	}
	if plan.CandidatesConsidered < 5 {
		t.Errorf("CandidatesConsidered = %d, want >= 5", plan.CandidatesConsidered)
	}
}

func TestPlannerRerank(t *testing.T) {
	t.Parallel()
	coll := newCollection(t)
	vecs := seedVectors(t, coll, 40)

	rr := &reverseRanker{}
	p, err := NewPlanner(coll, rr)
	if err != nil {
		t.Fatal(err)
	}

	res, plan, err := p.Run(context.Background(), vecs[0], Options{TopK: 3, OversearchFactor: 2.0, Rerank: true})
	if err != nil {
		t.Fatal(err)
	}
	if !plan.Reranked {
		t.Errorf("expected Reranked = true")
	}
	if len(res) != 3 {
		t.Errorf("len(res) = %d, want 3", len(res))
	}
	if !rr.called {
		t.Errorf("expected reranker to be invoked")
	}
}

func TestPlannerRerankWithoutReranker(t *testing.T) {
	t.Parallel()
	coll := newCollection(t)
	vecs := seedVectors(t, coll, 20)

	p, _ := NewPlanner(coll, nil)
	_, plan, err := p.Run(context.Background(), vecs[0], Options{TopK: 5, Rerank: true})
	if err != nil {
		t.Fatal(err)
	}
	if plan.Reranked {
		t.Errorf("expected Reranked = false when no reranker is configured")
	}
}

func TestPlannerCancellation(t *testing.T) {
	t.Parallel()
	coll := newCollection(t)
	seedVectors(t, coll, 10)

	p, _ := NewPlanner(coll, nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, _, err := p.Run(ctx, make([]float32, testDim), Options{TopK: 3})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}

func TestPlannerDimensionMismatch(t *testing.T) {
	t.Parallel()
	coll := newCollection(t)
	seedVectors(t, coll, 5)

	p, _ := NewPlanner(coll, nil)
	_, _, err := p.Run(context.Background(), make([]float32, testDim+1), Options{TopK: 3})
	if err == nil {
		t.Fatal("expected dim mismatch error")
	}
}

func TestTakeTopKSmall(t *testing.T) {
	t.Parallel()
	in := []index.SearchResult{
		{ID: "a", Score: 0.1},
		{ID: "b", Score: 0.5},
		{ID: "c", Score: 0.3},
	}
	out := takeTopK(in, 5)
	if len(out) != 3 {
		t.Fatalf("got len %d, want 3", len(out))
	}
	if out[0].ID != "b" || out[1].ID != "c" || out[2].ID != "a" {
		t.Errorf("unexpected order: %+v", out)
	}
}

func TestTakeTopKLarge(t *testing.T) {
	t.Parallel()
	in := []index.SearchResult{
		{ID: "a", Score: 0.1},
		{ID: "b", Score: 0.5},
		{ID: "c", Score: 0.3},
		{ID: "d", Score: 0.9},
		{ID: "e", Score: 0.2},
	}
	out := takeTopK(in, 2)
	if len(out) != 2 {
		t.Fatalf("got len %d, want 2", len(out))
	}
	if out[0].ID != "d" || out[1].ID != "b" {
		t.Errorf("unexpected order: %+v", out)
	}
}

// reverseRanker is a test reranker that flips the sign of all scores so the
// previously-worst candidate becomes the best.
type reverseRanker struct {
	called bool
}

func (r *reverseRanker) Rerank(_ []float32, candidates []index.SearchResult) ([]index.SearchResult, error) {
	r.called = true
	out := make([]index.SearchResult, len(candidates))
	for i, c := range candidates {
		c.Score = -c.Score
		out[i] = c
	}
	return out, nil
}
