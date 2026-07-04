package replication

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"
)

// fakeEngine records batched calls and can fail a configurable number of times.
type fakeEngine struct {
	inserts   [][]VectorRecord
	deletes   [][]string
	calls     []string // "upsert:<collection>:<n>" / "delete:<collection>:<n>" in order
	failNext  int
	failErr   error
	callCount int
}

func (f *fakeEngine) InsertBatch(_ context.Context, collection string, recs []VectorRecord) error {
	f.callCount++
	if f.failNext > 0 {
		f.failNext--
		return f.failErr
	}
	f.inserts = append(f.inserts, recs)
	f.calls = append(f.calls, fmt.Sprintf("upsert:%s:%d", collection, len(recs)))
	return nil
}

func (f *fakeEngine) DeleteBatch(_ context.Context, collection string, ids []string) error {
	f.callCount++
	if f.failNext > 0 {
		f.failNext--
		return f.failErr
	}
	f.deletes = append(f.deletes, ids)
	f.calls = append(f.calls, fmt.Sprintf("delete:%s:%d", collection, len(ids)))
	return nil
}

func noSleep(_ context.Context, _ time.Duration) error { return nil }

func upsertOp(collection, id string, lsn uint64) EngineOp {
	return EngineOp{Kind: EngineUpsert, Collection: collection, ID: id, Vector: []float32{1}, LSN: lsn}
}

func deleteOp(collection, id string, lsn uint64) EngineOp {
	return EngineOp{Kind: EngineDelete, Collection: collection, ID: id, LSN: lsn}
}

func TestWriterBatchesConsecutiveOpsAndPreservesOrder(t *testing.T) {
	fe := &fakeEngine{}
	w := NewWriter(fe, WriterConfig{Sleep: noSleep})
	ctx := context.Background()

	// upsert a, upsert b (same batch), delete a, upsert c — order matters.
	for _, op := range []EngineOp{
		upsertOp("docs", "a", 1),
		upsertOp("docs", "b", 2),
		deleteOp("docs", "a", 3),
		upsertOp("docs", "c", 4),
	} {
		if err := w.Apply(ctx, op); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Flush(ctx); err != nil {
		t.Fatal(err)
	}

	wantCalls := []string{"upsert:docs:2", "delete:docs:1", "upsert:docs:1"}
	if len(fe.calls) != len(wantCalls) {
		t.Fatalf("calls: got %v, want %v", fe.calls, wantCalls)
	}
	for i := range wantCalls {
		if fe.calls[i] != wantCalls[i] {
			t.Errorf("call %d: got %q, want %q", i, fe.calls[i], wantCalls[i])
		}
	}
	if w.AppliedLSN() != 4 {
		t.Errorf("AppliedLSN: got %d, want 4", w.AppliedLSN())
	}
}

func TestWriterAutoFlushAtMaxBatch(t *testing.T) {
	fe := &fakeEngine{}
	w := NewWriter(fe, WriterConfig{MaxBatch: 3, Sleep: noSleep})
	ctx := context.Background()

	for i := 0; i < 3; i++ {
		if err := w.Apply(ctx, upsertOp("docs", fmt.Sprintf("v%d", i), uint64(i+1))); err != nil {
			t.Fatal(err)
		}
	}
	// Batch limit hit: flush happened without an explicit call.
	if len(fe.inserts) != 1 || len(fe.inserts[0]) != 3 {
		t.Fatalf("auto-flush: inserts=%v", fe.inserts)
	}
	if w.AppliedLSN() != 3 {
		t.Errorf("AppliedLSN: got %d, want 3", w.AppliedLSN())
	}
}

func TestWriterRetriesTransientFailure(t *testing.T) {
	fe := &fakeEngine{failNext: 2, failErr: errors.New("transient")}
	w := NewWriter(fe, WriterConfig{MaxRetries: 4, Sleep: noSleep})
	ctx := context.Background()

	if err := w.Apply(ctx, upsertOp("docs", "a", 10)); err != nil {
		t.Fatal(err)
	}
	if err := w.Flush(ctx); err != nil {
		t.Fatalf("Flush should succeed after retries: %v", err)
	}
	if fe.callCount != 3 {
		t.Errorf("call count: got %d, want 3 (2 failures + 1 success)", fe.callCount)
	}
	if w.AppliedLSN() != 10 {
		t.Errorf("AppliedLSN: got %d, want 10", w.AppliedLSN())
	}
}

func TestWriterKeepsOpsOnFailedFlush(t *testing.T) {
	fe := &fakeEngine{failNext: 100, failErr: errors.New("down")}
	w := NewWriter(fe, WriterConfig{MaxRetries: 1, BreakerThreshold: 10, Sleep: noSleep})
	ctx := context.Background()

	if err := w.Apply(ctx, upsertOp("docs", "a", 5)); err != nil {
		t.Fatal(err)
	}
	if err := w.Flush(ctx); err == nil {
		t.Fatal("expected flush error")
	}
	if w.AppliedLSN() != 0 {
		t.Errorf("AppliedLSN must not advance on failure: got %d", w.AppliedLSN())
	}

	// Engine recovers; the same buffered op is delivered.
	fe.failNext = 0
	if err := w.Flush(ctx); err != nil {
		t.Fatalf("Flush after recovery: %v", err)
	}
	if len(fe.inserts) != 1 || fe.inserts[0][0].ID != "a" {
		t.Errorf("recovered flush: inserts=%v", fe.inserts)
	}
	if w.AppliedLSN() != 5 {
		t.Errorf("AppliedLSN: got %d, want 5", w.AppliedLSN())
	}
}

func TestWriterCircuitBreakerOpens(t *testing.T) {
	fe := &fakeEngine{failNext: 1000, failErr: errors.New("down")}
	w := NewWriter(fe, WriterConfig{MaxRetries: 1, BreakerThreshold: 3, Sleep: noSleep})
	ctx := context.Background()

	if err := w.Apply(ctx, upsertOp("docs", "a", 1)); err != nil {
		t.Fatal(err)
	}
	var lastErr error
	for i := 0; i < 3; i++ {
		lastErr = w.Flush(ctx)
		if lastErr == nil {
			t.Fatal("expected flush failure")
		}
	}
	if !errors.Is(lastErr, ErrCircuitOpen) {
		t.Fatalf("third failure should open circuit: %v", lastErr)
	}
	// Everything is rejected once open.
	if err := w.Apply(ctx, upsertOp("docs", "b", 2)); !errors.Is(err, ErrCircuitOpen) {
		t.Errorf("Apply after open: %v", err)
	}
	if err := w.Flush(ctx); !errors.Is(err, ErrCircuitOpen) {
		t.Errorf("Flush after open: %v", err)
	}
}

func TestWriterContextCancelDuringBackoff(t *testing.T) {
	fe := &fakeEngine{failNext: 1000, failErr: errors.New("down")}
	w := NewWriter(fe, WriterConfig{MaxRetries: 5, BaseBackoff: time.Hour})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if err := w.Apply(ctx, upsertOp("docs", "a", 1)); err != nil {
		t.Fatal(err)
	}
	err := w.Flush(ctx)
	if err == nil {
		t.Fatal("expected error")
	}
	// The hour-long backoff must have been interrupted by the context, not slept.
}

func TestWriterRejectsUnknownOpKind(t *testing.T) {
	w := NewWriter(&fakeEngine{}, WriterConfig{Sleep: noSleep})
	if err := w.Apply(context.Background(), EngineOp{Kind: 99}); err == nil {
		t.Error("expected error for unknown op kind")
	}
}

func TestWriterMultipleCollections(t *testing.T) {
	fe := &fakeEngine{}
	w := NewWriter(fe, WriterConfig{Sleep: noSleep})
	ctx := context.Background()

	// Alternating collections must not merge into one batch.
	if err := w.Apply(ctx, upsertOp("docs", "a", 1)); err != nil {
		t.Fatal(err)
	}
	if err := w.Apply(ctx, upsertOp("imgs", "b", 2)); err != nil {
		t.Fatal(err)
	}
	if err := w.Flush(ctx); err != nil {
		t.Fatal(err)
	}
	want := []string{"upsert:docs:1", "upsert:imgs:1"}
	if len(fe.calls) != 2 || fe.calls[0] != want[0] || fe.calls[1] != want[1] {
		t.Errorf("calls: got %v, want %v", fe.calls, want)
	}
}
