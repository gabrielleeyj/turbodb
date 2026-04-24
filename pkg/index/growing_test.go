package index

import (
	"fmt"
	"math/rand/v2"
	"sync"
	"testing"

	"github.com/gabrielleeyj/turbodb/pkg/codebook"
	"github.com/gabrielleeyj/turbodb/pkg/rotation"
)

const (
	testDim      = 128
	testBitWidth = 4
	testSeed     = 42
)

func newTestGrowingSegment(t *testing.T) *GrowingSegment {
	t.Helper()
	rot, err := rotation.NewHadamardRotator(testDim, testSeed)
	if err != nil {
		t.Fatalf("NewHadamardRotator: %v", err)
	}
	cb, err := codebook.Load(testDim, testBitWidth)
	if err != nil {
		t.Fatalf("codebook.Load: %v", err)
	}
	seg, err := NewGrowingSegment(GrowingSegmentConfig{
		ID:       "test-seg-0001",
		Dim:      testDim,
		Rotator:  rot,
		Codebook: cb,
		BitWidth: testBitWidth,
	})
	if err != nil {
		t.Fatalf("NewGrowingSegment: %v", err)
	}
	return seg
}

func randomVec(rng *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = float32(rng.NormFloat64())
	}
	return v
}

func TestGrowingSegmentInsertAndCount(t *testing.T) {
	seg := newTestGrowingSegment(t)
	rng := rand.New(rand.NewPCG(1, 2))

	for i := range 100 {
		err := seg.Insert(VectorEntry{
			ID:     fmt.Sprintf("vec-%d", i),
			Values: randomVec(rng, testDim),
		})
		if err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}

	if seg.Count() != 100 {
		t.Fatalf("Count: got %d, want 100", seg.Count())
	}
}

func TestGrowingSegmentDuplicateID(t *testing.T) {
	seg := newTestGrowingSegment(t)
	rng := rand.New(rand.NewPCG(1, 2))

	v := randomVec(rng, testDim)
	if err := seg.Insert(VectorEntry{ID: "dup", Values: v}); err != nil {
		t.Fatalf("first insert: %v", err)
	}

	err := seg.Insert(VectorEntry{ID: "dup", Values: v})
	if err == nil {
		t.Fatal("expected error for duplicate ID")
	}
}

func TestGrowingSegmentDimMismatch(t *testing.T) {
	seg := newTestGrowingSegment(t)
	err := seg.Insert(VectorEntry{ID: "bad", Values: []float32{1, 2, 3}})
	if err == nil {
		t.Fatal("expected error for dimension mismatch")
	}
}

func TestGrowingSegmentEmptyID(t *testing.T) {
	seg := newTestGrowingSegment(t)
	err := seg.Insert(VectorEntry{ID: "", Values: make([]float32, testDim)})
	if err == nil {
		t.Fatal("expected error for empty ID")
	}
}

func TestGrowingSegmentContains(t *testing.T) {
	seg := newTestGrowingSegment(t)
	rng := rand.New(rand.NewPCG(1, 2))

	if err := seg.Insert(VectorEntry{ID: "exists", Values: randomVec(rng, testDim)}); err != nil {
		t.Fatal(err)
	}

	if !seg.Contains("exists") {
		t.Fatal("expected Contains to return true")
	}
	if seg.Contains("missing") {
		t.Fatal("expected Contains to return false")
	}
}

func TestGrowingSegmentSearch(t *testing.T) {
	seg := newTestGrowingSegment(t)
	rng := rand.New(rand.NewPCG(1, 2))

	// Insert 50 vectors.
	for i := range 50 {
		err := seg.Insert(VectorEntry{
			ID:     fmt.Sprintf("vec-%d", i),
			Values: randomVec(rng, testDim),
		})
		if err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}

	query := randomVec(rng, testDim)
	results, err := seg.Search(query, 5, nil)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}

	if len(results) != 5 {
		t.Fatalf("Search returned %d results, want 5", len(results))
	}

	// Verify descending order.
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Fatalf("results not in descending order at index %d: %f > %f",
				i, results[i].Score, results[i-1].Score)
		}
	}
}

func TestGrowingSegmentSearchWithTombstones(t *testing.T) {
	seg := newTestGrowingSegment(t)
	rng := rand.New(rand.NewPCG(1, 2))

	for i := range 10 {
		err := seg.Insert(VectorEntry{
			ID:     fmt.Sprintf("vec-%d", i),
			Values: randomVec(rng, testDim),
		})
		if err != nil {
			t.Fatal(err)
		}
	}

	tombstones := NewTombstoneLog()
	tombstones.Delete("vec-0")
	tombstones.Delete("vec-1")

	query := randomVec(rng, testDim)
	results, err := seg.Search(query, 20, tombstones)
	if err != nil {
		t.Fatal(err)
	}

	// Should get 8 results (10 - 2 tombstoned).
	if len(results) != 8 {
		t.Fatalf("expected 8 results, got %d", len(results))
	}

	for _, r := range results {
		if r.ID == "vec-0" || r.ID == "vec-1" {
			t.Fatalf("tombstoned ID %q appeared in results", r.ID)
		}
	}
}

func TestGrowingSegmentSearchTopKGreaterThanCount(t *testing.T) {
	seg := newTestGrowingSegment(t)
	rng := rand.New(rand.NewPCG(1, 2))

	for i := range 3 {
		if err := seg.Insert(VectorEntry{ID: fmt.Sprintf("v%d", i), Values: randomVec(rng, testDim)}); err != nil {
			t.Fatal(err)
		}
	}

	results, err := seg.Search(randomVec(rng, testDim), 10, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}
}

func TestGrowingSegmentInfo(t *testing.T) {
	seg := newTestGrowingSegment(t)
	info := seg.Info()

	if info.ID != "test-seg-0001" {
		t.Fatalf("ID: got %q, want %q", info.ID, "test-seg-0001")
	}
	if info.Type != SegmentTypeGrowing {
		t.Fatalf("Type: got %v, want Growing", info.Type)
	}
	if info.Dim != testDim {
		t.Fatalf("Dim: got %d, want %d", info.Dim, testDim)
	}
}

func TestGrowingSegmentConcurrentInserts(t *testing.T) {
	seg := newTestGrowingSegment(t)

	var wg sync.WaitGroup
	errs := make(chan error, 100)

	for i := range 100 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rng := rand.New(rand.NewPCG(uint64(i), uint64(i+1000)))
			err := seg.Insert(VectorEntry{
				ID:     fmt.Sprintf("concurrent-%d", i),
				Values: randomVec(rng, testDim),
			})
			if err != nil {
				errs <- err
			}
		}()
	}

	wg.Wait()
	close(errs)

	for err := range errs {
		t.Fatalf("concurrent insert error: %v", err)
	}

	if seg.Count() != 100 {
		t.Fatalf("Count: got %d, want 100", seg.Count())
	}
}

func TestGrowingSegmentImmutableStorage(t *testing.T) {
	seg := newTestGrowingSegment(t)

	original := make([]float32, testDim)
	for i := range original {
		original[i] = float32(i)
	}
	if err := seg.Insert(VectorEntry{ID: "immut", Values: original}); err != nil {
		t.Fatal(err)
	}

	// Mutate the original slice.
	original[0] = 999.0

	entries := seg.Entries()
	if entries[0].Values[0] == 999.0 {
		t.Fatal("stored values were mutated — immutability violated")
	}
}
