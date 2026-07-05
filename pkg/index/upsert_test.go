package index

import (
	"math/rand/v2"
	"slices"
	"testing"
)

func upsertVec(vals ...float32) []float32 {
	v := make([]float32, testDim)
	copy(v, vals)
	return v
}

func TestCollectionUpsertReplacesInGrowing(t *testing.T) {
	c := newTestCollection(t, 1000)

	if err := c.Upsert(VectorEntry{ID: "a", Values: upsertVec(1)}); err != nil {
		t.Fatal(err)
	}
	if err := c.Upsert(VectorEntry{ID: "a", Values: upsertVec(2)}); err != nil {
		t.Fatalf("upsert of existing id must succeed: %v", err)
	}

	if got := c.Stats().VectorCount; got != 1 {
		t.Errorf("vector count: got %d, want 1", got)
	}
	snap, err := c.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if len(snap) != 1 || snap[0].Values[0] != 2 {
		t.Errorf("snapshot: %+v (want single entry with replaced value)", snap)
	}
}

func TestCollectionDeleteIsPhysicalInGrowing(t *testing.T) {
	c := newTestCollection(t, 1000)

	if err := c.Upsert(VectorEntry{ID: "a", Values: upsertVec(1)}); err != nil {
		t.Fatal(err)
	}
	if err := c.Delete("a"); err != nil {
		t.Fatal(err)
	}
	if got := c.Stats().VectorCount; got != 0 {
		t.Errorf("vector count after delete: got %d, want 0", got)
	}
	if c.Stats().TombstoneCount != 0 {
		t.Error("growing-only delete must not leave a tombstone")
	}
	if len(c.IDs()) != 0 {
		t.Errorf("ids after delete: %v", c.IDs())
	}
	// Re-upsert after delete is visible again.
	if err := c.Upsert(VectorEntry{ID: "a", Values: upsertVec(3)}); err != nil {
		t.Fatal(err)
	}
	if got := c.IDs(); len(got) != 1 || got[0] != "a" {
		t.Errorf("ids after re-upsert: %v", got)
	}
}

func TestCollectionUpsertOfSealedResidentID(t *testing.T) {
	// Threshold 3 so the first three vectors seal.
	c := newTestCollection(t, 3)
	rng := rand.New(rand.NewPCG(3, 4))
	for _, id := range []string{"s1", "s2", "s3"} {
		if err := c.Upsert(VectorEntry{ID: id, Values: randomVec(rng)}); err != nil {
			t.Fatal(err)
		}
	}
	if err := c.Flush(); err != nil {
		t.Fatal(err)
	}
	if c.Stats().SealedSegmentCount == 0 {
		t.Fatal("expected a sealed segment")
	}

	// Upsert a sealed-resident id: must succeed, id visible exactly once.
	if err := c.Upsert(VectorEntry{ID: "s2", Values: randomVec(rng)}); err != nil {
		t.Fatalf("upsert of sealed-resident id: %v", err)
	}
	ids := c.IDs()
	if !slices.Equal(ids, []string{"s1", "s2", "s3"}) {
		t.Errorf("ids: %v, want [s1 s2 s3]", ids)
	}
	results, err := c.Search(randomVec(rng), 10)
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]int{}
	for _, r := range results {
		seen[r.ID]++
	}
	if seen["s2"] != 1 {
		t.Errorf("s2 must appear exactly once in search results: %v", results)
	}

	// Delete the upserted id: gone from both worlds.
	if err := c.Delete("s2"); err != nil {
		t.Fatal(err)
	}
	if slices.Contains(c.IDs(), "s2") {
		t.Errorf("s2 still visible after delete: %v", c.IDs())
	}
}

func TestCollectionSealAfterSealedUpsertKeepsIDVisible(t *testing.T) {
	c := newTestCollection(t, 3)
	rng := rand.New(rand.NewPCG(5, 6))
	for _, id := range []string{"a", "b", "c"} {
		if err := c.Upsert(VectorEntry{ID: id, Values: randomVec(rng)}); err != nil {
			t.Fatal(err)
		}
	}
	if err := c.Flush(); err != nil {
		t.Fatal(err)
	}

	// Upsert "a" (now sealed-resident), then force the growing segment to
	// seal as well: the tombstone masking the old copy must be lifted so
	// the new copy stays visible, and search must return "a" exactly once.
	if err := c.Upsert(VectorEntry{ID: "a", Values: randomVec(rng)}); err != nil {
		t.Fatal(err)
	}
	if err := c.Flush(); err != nil {
		t.Fatal(err)
	}

	if !slices.Contains(c.IDs(), "a") {
		t.Fatalf("a lost after double seal: %v", c.IDs())
	}
	results, err := c.Search(randomVec(rng), 10)
	if err != nil {
		t.Fatal(err)
	}
	count := 0
	for _, r := range results {
		if r.ID == "a" {
			count++
		}
	}
	if count != 1 {
		t.Errorf("a must appear exactly once after double seal, got %d (%v)", count, results)
	}
}
