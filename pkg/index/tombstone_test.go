package index

import (
	"sync"
	"testing"
)

func TestTombstoneLogBasic(t *testing.T) {
	tl := NewTombstoneLog()

	if tl.Count() != 0 {
		t.Fatalf("Count: got %d, want 0", tl.Count())
	}

	tl.Delete("a")
	tl.Delete("b")

	if tl.Count() != 2 {
		t.Fatalf("Count: got %d, want 2", tl.Count())
	}

	if !tl.IsDeleted("a") {
		t.Fatal("expected 'a' to be deleted")
	}
	if tl.IsDeleted("c") {
		t.Fatal("expected 'c' to not be deleted")
	}
}

func TestTombstoneLogIDs(t *testing.T) {
	tl := NewTombstoneLogFrom([]string{"x", "y", "z"})

	ids := tl.IDs()
	if len(ids) != 3 {
		t.Fatalf("IDs: got %d, want 3", len(ids))
	}

	idSet := make(map[string]bool)
	for _, id := range ids {
		idSet[id] = true
	}
	for _, expected := range []string{"x", "y", "z"} {
		if !idSet[expected] {
			t.Fatalf("missing expected ID %q", expected)
		}
	}
}

func TestTombstoneLogRemove(t *testing.T) {
	tl := NewTombstoneLog()
	tl.Delete("a")
	tl.Delete("b")

	tl.Remove("a")
	if tl.IsDeleted("a") {
		t.Fatal("'a' should no longer be deleted")
	}
	if tl.Count() != 1 {
		t.Fatalf("Count: got %d, want 1", tl.Count())
	}
}

func TestTombstoneLogClear(t *testing.T) {
	tl := NewTombstoneLogFrom([]string{"a", "b", "c"})
	tl.Clear()

	if tl.Count() != 0 {
		t.Fatalf("Count after clear: got %d, want 0", tl.Count())
	}
}

func TestTombstoneLogConcurrent(t *testing.T) {
	tl := NewTombstoneLog()
	var wg sync.WaitGroup

	// Concurrent writes.
	for i := range 100 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			tl.Delete(string(rune('A' + i%26)))
		}()
	}

	// Concurrent reads.
	for range 100 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			tl.IsDeleted("A")
			tl.Count()
		}()
	}

	wg.Wait()
}
