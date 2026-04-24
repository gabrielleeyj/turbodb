package index

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/gabrielleeyj/turbodb/pkg/codebook"
	"github.com/gabrielleeyj/turbodb/pkg/rotation"
)

func newTestSealedSegment(t *testing.T, n int) *SealedSegment {
	t.Helper()
	rot, err := rotation.NewHadamardRotator(testDim, testSeed)
	if err != nil {
		t.Fatalf("NewHadamardRotator: %v", err)
	}
	cb, err := codebook.Load(testDim, testBitWidth)
	if err != nil {
		t.Fatalf("codebook.Load: %v", err)
	}

	rng := rand.New(rand.NewPCG(10, 20))
	entries := make([]VectorEntry, n)
	for i := range n {
		entries[i] = VectorEntry{
			ID:     fmt.Sprintf("sealed-vec-%d", i),
			Values: randomVec(rng, testDim),
		}
	}

	seg, err := Seal("sealed-001", entries, SealedSegmentConfig{
		ID:       "sealed-001",
		Dim:      testDim,
		BitWidth: testBitWidth,
		Rotator:  rot,
		Codebook: cb,
	})
	if err != nil {
		t.Fatalf("Seal: %v", err)
	}
	return seg
}

func TestSealBasic(t *testing.T) {
	seg := newTestSealedSegment(t, 100)

	if seg.Count() != 100 {
		t.Fatalf("Count: got %d, want 100", seg.Count())
	}
	if seg.Type() != SegmentTypeSealed {
		t.Fatalf("Type: got %v, want Sealed", seg.Type())
	}
	if seg.ID() != "sealed-001" {
		t.Fatalf("ID: got %q, want %q", seg.ID(), "sealed-001")
	}
}

func TestSealedSegmentContains(t *testing.T) {
	seg := newTestSealedSegment(t, 10)

	if !seg.Contains("sealed-vec-0") {
		t.Fatal("expected Contains to return true for sealed-vec-0")
	}
	if seg.Contains("nonexistent") {
		t.Fatal("expected Contains to return false for nonexistent")
	}
}

func TestSealedSegmentSearch(t *testing.T) {
	seg := newTestSealedSegment(t, 100)
	rng := rand.New(rand.NewPCG(99, 100))
	query := randomVec(rng, testDim)

	results, err := seg.Search(query, 5, nil)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}

	if len(results) != 5 {
		t.Fatalf("expected 5 results, got %d", len(results))
	}

	// Verify descending order.
	for i := 1; i < len(results); i++ {
		if results[i].Score > results[i-1].Score {
			t.Fatalf("results not in descending order at %d: %f > %f",
				i, results[i].Score, results[i-1].Score)
		}
	}
}

func TestSealedSegmentSearchWithTombstones(t *testing.T) {
	seg := newTestSealedSegment(t, 20)
	rng := rand.New(rand.NewPCG(99, 100))
	query := randomVec(rng, testDim)

	tombstones := NewTombstoneLog()
	tombstones.Delete("sealed-vec-0")
	tombstones.Delete("sealed-vec-5")
	tombstones.Delete("sealed-vec-10")

	results, err := seg.Search(query, 50, tombstones)
	if err != nil {
		t.Fatal(err)
	}

	if len(results) != 17 {
		t.Fatalf("expected 17 results (20 - 3 tombstoned), got %d", len(results))
	}

	for _, r := range results {
		if tombstones.IsDeleted(r.ID) {
			t.Fatalf("tombstoned ID %q in results", r.ID)
		}
	}
}

func TestSealedSegmentSearchRecallVsGrowing(t *testing.T) {
	// Verify that sealed segment search returns similar results to brute-force.
	rot, err := rotation.NewHadamardRotator(testDim, testSeed)
	if err != nil {
		t.Fatal(err)
	}
	cb, err := codebook.Load(testDim, testBitWidth)
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewPCG(55, 66))
	entries := make([]VectorEntry, 200)
	for i := range 200 {
		entries[i] = VectorEntry{
			ID:     fmt.Sprintf("v%d", i),
			Values: randomVec(rng, testDim),
		}
	}

	// Build growing segment with same vectors.
	growing, err := NewGrowingSegment(GrowingSegmentConfig{
		ID: "growing-recall", Dim: testDim,
		Rotator: rot, Codebook: cb, BitWidth: testBitWidth,
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		if err := growing.Insert(e); err != nil {
			t.Fatal(err)
		}
	}

	// Build sealed segment.
	sealed, err := Seal("sealed-recall", entries, SealedSegmentConfig{
		ID: "sealed-recall", Dim: testDim, BitWidth: testBitWidth,
		Rotator: rot, Codebook: cb,
	})
	if err != nil {
		t.Fatal(err)
	}

	query := randomVec(rng, testDim)
	topK := 10
	growingResults, _ := growing.Search(query, topK, nil)
	sealedResults, _ := sealed.Search(query, topK, nil)

	// Measure recall: how many of the true top-K appear in the sealed results.
	trueIDs := make(map[string]bool, topK)
	for _, r := range growingResults {
		trueIDs[r.ID] = true
	}

	var hits int
	for _, r := range sealedResults {
		if trueIDs[r.ID] {
			hits++
		}
	}

	recall := float64(hits) / float64(topK)
	t.Logf("Recall@%d: %.2f (%d/%d hits)", topK, recall, hits, topK)

	// At 4-bit quantization with d=128, recall should be reasonable.
	if recall < 0.5 {
		t.Fatalf("recall too low: %.2f (expected >= 0.5)", recall)
	}
}

func TestSealEmptyEntries(t *testing.T) {
	rot, _ := rotation.NewHadamardRotator(testDim, testSeed)
	cb, _ := codebook.Load(testDim, testBitWidth)

	_, err := Seal("empty", nil, SealedSegmentConfig{
		ID: "empty", Dim: testDim, BitWidth: testBitWidth,
		Rotator: rot, Codebook: cb,
	})
	if err == nil {
		t.Fatal("expected error for empty entries")
	}
}

func TestSealedSegmentInfo(t *testing.T) {
	seg := newTestSealedSegment(t, 10)
	info := seg.Info()

	if info.Type != SegmentTypeSealed {
		t.Fatalf("Type: got %v, want Sealed", info.Type)
	}
	if info.Count != 10 {
		t.Fatalf("Count: got %d, want 10", info.Count)
	}
	if info.SealedAt.IsZero() {
		t.Fatal("SealedAt should not be zero")
	}
}
