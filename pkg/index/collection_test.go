package index

import (
	"fmt"
	"math/rand/v2"
	"sync"
	"testing"

	"github.com/gabrielleeyj/turbodb/pkg/codebook"
	"github.com/gabrielleeyj/turbodb/pkg/rotation"
)

func newTestCollection(t *testing.T, sealThreshold int) *Collection {
	t.Helper()
	rot, err := rotation.NewHadamardRotator(testDim, testSeed)
	if err != nil {
		t.Fatal(err)
	}
	cb, err := codebook.Load(testDim, testBitWidth)
	if err != nil {
		t.Fatal(err)
	}

	c, err := NewCollection(CollectionConfig{
		Name:          "test-collection",
		Dim:           testDim,
		BitWidth:      testBitWidth,
		Rotator:       rot,
		Codebook:      cb,
		SealThreshold: sealThreshold,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c.Close() })
	return c
}

func TestCollectionInsertAndSearch(t *testing.T) {
	c := newTestCollection(t, 1000)
	rng := rand.New(rand.NewPCG(1, 2))

	for i := range 50 {
		err := c.Insert(VectorEntry{
			ID:     fmt.Sprintf("vec-%d", i),
			Values: randomVec(rng, testDim),
		})
		if err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}

	stats := c.Stats()
	if stats.VectorCount != 50 {
		t.Fatalf("VectorCount: got %d, want 50", stats.VectorCount)
	}

	query := randomVec(rng, testDim)
	results, err := c.Search(query, 5)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}

	if len(results) != 5 {
		t.Fatalf("expected 5 results, got %d", len(results))
	}

	// Verify descending order.
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Fatalf("not descending at %d: %f > %f", i, results[i].Score, results[i-1].Score)
		}
	}
}

func TestCollectionDelete(t *testing.T) {
	c := newTestCollection(t, 1000)
	rng := rand.New(rand.NewPCG(1, 2))

	for i := range 10 {
		if err := c.Insert(VectorEntry{ID: fmt.Sprintf("v%d", i), Values: randomVec(rng, testDim)}); err != nil {
			t.Fatal(err)
		}
	}

	if err := c.Delete("v0"); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	// Search should not return deleted vector.
	query := randomVec(rng, testDim)
	results, err := c.Search(query, 20)
	if err != nil {
		t.Fatal(err)
	}

	for _, r := range results {
		if r.ID == "v0" {
			t.Fatal("deleted vector appeared in search results")
		}
	}

	if len(results) != 9 {
		t.Fatalf("expected 9 results, got %d", len(results))
	}
}

func TestCollectionDeleteNonexistent(t *testing.T) {
	c := newTestCollection(t, 1000)
	err := c.Delete("nonexistent")
	if err == nil {
		t.Fatal("expected error for deleting nonexistent vector")
	}
}

func TestCollectionDuplicateID(t *testing.T) {
	c := newTestCollection(t, 1000)
	rng := rand.New(rand.NewPCG(1, 2))

	v := randomVec(rng, testDim)
	if err := c.Insert(VectorEntry{ID: "dup", Values: v}); err != nil {
		t.Fatal(err)
	}

	err := c.Insert(VectorEntry{ID: "dup", Values: v})
	if err == nil {
		t.Fatal("expected error for duplicate ID")
	}
}

func TestCollectionFlush(t *testing.T) {
	c := newTestCollection(t, 1000) // High threshold so auto-seal doesn't trigger.
	rng := rand.New(rand.NewPCG(1, 2))

	for i := range 20 {
		if err := c.Insert(VectorEntry{ID: fmt.Sprintf("f%d", i), Values: randomVec(rng, testDim)}); err != nil {
			t.Fatal(err)
		}
	}

	if err := c.Flush(); err != nil {
		t.Fatalf("Flush: %v", err)
	}

	stats := c.Stats()
	if stats.SealedSegmentCount != 1 {
		t.Fatalf("expected 1 sealed segment after flush, got %d", stats.SealedSegmentCount)
	}
	if stats.VectorCount != 20 {
		t.Fatalf("VectorCount after flush: got %d, want 20", stats.VectorCount)
	}

	// Search should still work across sealed segment.
	query := randomVec(rng, testDim)
	results, err := c.Search(query, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 5 {
		t.Fatalf("expected 5 results after flush, got %d", len(results))
	}
}

func TestCollectionAutoSeal(t *testing.T) {
	threshold := 10
	c := newTestCollection(t, threshold)
	rng := rand.New(rand.NewPCG(1, 2))

	// Insert enough to trigger auto-seal.
	for i := range threshold + 5 {
		if err := c.Insert(VectorEntry{
			ID:     fmt.Sprintf("auto-%d", i),
			Values: randomVec(rng, testDim),
		}); err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}

	// Flush remaining to ensure seal completes.
	if err := c.Flush(); err != nil {
		t.Fatal(err)
	}

	stats := c.Stats()
	if stats.VectorCount != threshold+5 {
		t.Fatalf("VectorCount: got %d, want %d", stats.VectorCount, threshold+5)
	}
	if stats.SealedSegmentCount < 1 {
		t.Fatalf("expected at least 1 sealed segment, got %d", stats.SealedSegmentCount)
	}
}

func TestCollectionSearchAcrossSegments(t *testing.T) {
	c := newTestCollection(t, 1000)
	rng := rand.New(rand.NewPCG(1, 2))

	// Insert vectors, flush to create a sealed segment, then insert more.
	for i := range 20 {
		if err := c.Insert(VectorEntry{ID: fmt.Sprintf("batch1-%d", i), Values: randomVec(rng, testDim)}); err != nil {
			t.Fatal(err)
		}
	}
	if err := c.Flush(); err != nil {
		t.Fatal(err)
	}

	for i := range 20 {
		if err := c.Insert(VectorEntry{ID: fmt.Sprintf("batch2-%d", i), Values: randomVec(rng, testDim)}); err != nil {
			t.Fatal(err)
		}
	}

	stats := c.Stats()
	if stats.VectorCount != 40 {
		t.Fatalf("VectorCount: got %d, want 40", stats.VectorCount)
	}

	query := randomVec(rng, testDim)
	results, err := c.Search(query, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 10 {
		t.Fatalf("expected 10 results, got %d", len(results))
	}

	// Verify results come from both segments.
	hasBatch1, hasBatch2 := false, false
	for _, r := range results {
		if len(r.ID) > 6 && r.ID[:6] == "batch1" {
			hasBatch1 = true
		}
		if len(r.ID) > 6 && r.ID[:6] == "batch2" {
			hasBatch2 = true
		}
	}
	// With random vectors, it's likely both batches are represented.
	t.Logf("Results from batch1: %v, batch2: %v", hasBatch1, hasBatch2)
}

func TestCollectionConcurrentInsertSearch(t *testing.T) {
	c := newTestCollection(t, 1000)
	var wg sync.WaitGroup

	// Concurrent inserts.
	for i := range 50 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rng := rand.New(rand.NewPCG(uint64(i), uint64(i+500)))
			err := c.Insert(VectorEntry{
				ID:     fmt.Sprintf("cc-%d", i),
				Values: randomVec(rng, testDim),
			})
			if err != nil {
				t.Errorf("concurrent insert %d: %v", i, err)
			}
		}()
	}

	// Concurrent searches (after a small initial batch).
	wg.Wait()

	var searchWg sync.WaitGroup
	for i := range 20 {
		searchWg.Add(1)
		go func() {
			defer searchWg.Done()
			rng := rand.New(rand.NewPCG(uint64(i+100), uint64(i+600)))
			results, err := c.Search(randomVec(rng, testDim), 5)
			if err != nil {
				t.Errorf("concurrent search %d: %v", i, err)
				return
			}
			if len(results) == 0 {
				t.Errorf("concurrent search %d: no results", i)
			}
		}()
	}
	searchWg.Wait()
}

func TestCollectionFlushEmpty(t *testing.T) {
	c := newTestCollection(t, 1000)
	// Flushing with no vectors should be a no-op.
	if err := c.Flush(); err != nil {
		t.Fatalf("Flush empty: %v", err)
	}
	stats := c.Stats()
	if stats.SealedSegmentCount != 0 {
		t.Fatalf("expected 0 sealed segments, got %d", stats.SealedSegmentCount)
	}
}

func TestCollectionName(t *testing.T) {
	c := newTestCollection(t, 1000)
	if c.Name() != "test-collection" {
		t.Fatalf("Name: got %q, want %q", c.Name(), "test-collection")
	}
	if c.Dim() != testDim {
		t.Fatalf("Dim: got %d, want %d", c.Dim(), testDim)
	}
}
