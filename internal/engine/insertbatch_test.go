package engine

import (
	"context"
	"fmt"
	"math/rand/v2"
	"path/filepath"
	"testing"

	"github.com/gabrielleeyj/turbodb/pkg/index"
)

func TestEngineInsertBatch(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	e, err := New(Config{DataDir: filepath.Join(dir, "data")})
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	if err := e.CreateCollection(ctx, defaultCollection("batch")); err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewPCG(5, 6))
	entries := make([]index.VectorEntry, 50)
	for i := range entries {
		entries[i] = index.VectorEntry{ID: fmt.Sprintf("b%02d", i), Values: randVec(rng)}
	}
	n, err := e.InsertBatch(ctx, "batch", entries)
	if err != nil || n != 50 {
		t.Fatalf("InsertBatch: n=%d err=%v", n, err)
	}

	// All-or-nothing validation: one bad dim rejects the whole batch.
	bad := append([]index.VectorEntry{}, entries[:2]...)
	bad[1] = index.VectorEntry{ID: "short", Values: []float32{1, 2}}
	if _, err := e.InsertBatch(ctx, "batch", bad); err == nil {
		t.Error("expected dim validation error")
	}

	// Durability: reopen and confirm WAL replay restores the batch.
	if err := e.Close(); err != nil {
		t.Fatal(err)
	}
	e2, err := New(Config{DataDir: filepath.Join(dir, "data")})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = e2.Close() }()
	ids, err := e2.ListIDs("batch")
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 50 {
		t.Errorf("after replay: %d ids, want 50", len(ids))
	}

	if _, err := e2.InsertBatch(ctx, "nope", entries); err == nil {
		t.Error("expected error for unknown collection")
	}
	if n, err := e2.InsertBatch(ctx, "batch", nil); n != 0 || err != nil {
		t.Errorf("empty batch: n=%d err=%v", n, err)
	}
}

func TestEngineInsertBatchIsIdempotent(t *testing.T) {
	t.Parallel()
	e := newTestEngine(t)
	ctx := context.Background()
	if err := e.CreateCollection(ctx, defaultCollection("redeliver")); err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewPCG(9, 9))
	entries := make([]index.VectorEntry, 10)
	for i := range entries {
		entries[i] = index.VectorEntry{ID: fmt.Sprintf("r%02d", i), Values: randVec(rng)}
	}

	// At-least-once CDC delivery replays batches after a crash: the second
	// delivery must succeed and not duplicate anything.
	for round := 0; round < 2; round++ {
		if _, err := e.InsertBatch(ctx, "redeliver", entries); err != nil {
			t.Fatalf("round %d: %v", round, err)
		}
	}
	ids, err := e.ListIDs("redeliver")
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 10 {
		t.Errorf("ids after redelivery: got %d, want 10", len(ids))
	}

	// Redelivered deletes are equally idempotent.
	if err := e.Delete(ctx, "redeliver", "r00"); err != nil {
		t.Fatal(err)
	}
	if err := e.Delete(ctx, "redeliver", "r00"); err != nil {
		t.Errorf("second delete must be a no-op success: %v", err)
	}
	ids, _ = e.ListIDs("redeliver")
	if len(ids) != 9 {
		t.Errorf("ids after delete: got %d, want 9", len(ids))
	}
}
