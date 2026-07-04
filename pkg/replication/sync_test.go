package replication

import (
	"context"
	"errors"
	"io"
	"path/filepath"
	"testing"
	"time"
)

// sliceSource replays a fixed set of events then returns io.EOF.
type sliceSource struct {
	events []ChangeEvent
	pos    int
}

func (s *sliceSource) Next(ctx context.Context) (ChangeEvent, error) {
	if ctx.Err() != nil {
		return ChangeEvent{}, ctx.Err()
	}
	if s.pos >= len(s.events) {
		return ChangeEvent{}, io.EOF
	}
	ev := s.events[s.pos]
	s.pos++
	return ev, nil
}

func (s *sliceSource) Close() error { return nil }

func newSyncFixtures(t *testing.T, fe EngineClient) (*Transformer, *Writer, *FileCheckpoint) {
	t.Helper()
	cfg, err := ParseConfig([]byte(validYAML))
	if err != nil {
		t.Fatal(err)
	}
	cp, err := NewFileCheckpoint(filepath.Join(t.TempDir(), "sync.ckpt"))
	if err != nil {
		t.Fatal(err)
	}
	return NewTransformer(cfg), NewWriter(fe, WriterConfig{Sleep: noSleep}), cp
}

func TestSyncEndToEnd(t *testing.T) {
	fe := &fakeEngine{}
	tr, w, cp := newSyncFixtures(t, fe)

	src := &sliceSource{events: []ChangeEvent{
		{Op: OpInsert, Table: "public.documents", LSN: 1,
			Row: map[string]any{"doc_id": "a", "vector": "[1,2]"}},
		{Op: OpInsert, Table: "public.ignored", LSN: 2,
			Row: map[string]any{"x": "y"}},
		{Op: OpUpdate, Table: "public.documents", LSN: 3,
			Row: map[string]any{"doc_id": "a", "vector": "[1,2]", "deleted_at": "2026-01-01"}},
		{Op: OpInsert, Table: "public.documents", LSN: 4,
			Row: map[string]any{"doc_id": "bad"}}, // malformed: no vector — skipped
		{Op: OpDelete, Table: "public.documents", LSN: 5,
			Row: map[string]any{"doc_id": "z"}},
	}}

	if err := Sync(context.Background(), src, tr, w, cp, SyncOptions{}); err != nil {
		t.Fatalf("Sync: %v", err)
	}

	// a upserted; then soft-delete update -> delete a; delete z (merged batch).
	want := []string{"upsert:docs:1", "delete:docs:2"}
	if len(fe.calls) != len(want) || fe.calls[0] != want[0] || fe.calls[1] != want[1] {
		t.Errorf("calls: got %v, want %v", fe.calls, want)
	}

	// Checkpoint reflects the highest applied LSN.
	lsn, err := cp.Load()
	if err != nil {
		t.Fatal(err)
	}
	if lsn != 5 {
		t.Errorf("checkpoint: got %d, want 5", lsn)
	}
}

func TestSyncStopsOnCircuitOpen(t *testing.T) {
	fe := &fakeEngine{failNext: 1000, failErr: errors.New("engine down")}
	cfg, err := ParseConfig([]byte(validYAML))
	if err != nil {
		t.Fatal(err)
	}
	cp, err := NewFileCheckpoint(filepath.Join(t.TempDir(), "sync.ckpt"))
	if err != nil {
		t.Fatal(err)
	}
	w := NewWriter(fe, WriterConfig{MaxRetries: 1, BreakerThreshold: 1, MaxBatch: 1, Sleep: noSleep})

	src := &sliceSource{events: []ChangeEvent{
		{Op: OpInsert, Table: "public.documents", LSN: 1,
			Row: map[string]any{"doc_id": "a", "vector": "[1]"}},
	}}
	err = Sync(context.Background(), src, NewTransformer(cfg), w, cp, SyncOptions{})
	if !errors.Is(err, ErrCircuitOpen) {
		t.Fatalf("Sync: got %v, want ErrCircuitOpen", err)
	}
	// Nothing acked.
	if lsn, _ := cp.Load(); lsn != 0 {
		t.Errorf("checkpoint must stay 0 on failure, got %d", lsn)
	}
}

func TestSyncHonorsContextCancellation(t *testing.T) {
	fe := &fakeEngine{}
	tr, w, cp := newSyncFixtures(t, fe)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	src := &sliceSource{events: []ChangeEvent{
		{Op: OpInsert, Table: "public.documents", LSN: 1,
			Row: map[string]any{"doc_id": "a", "vector": "[1]"}},
	}}
	err := Sync(ctx, src, tr, w, cp, SyncOptions{FlushInterval: time.Hour})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Sync: got %v, want context.Canceled", err)
	}
}
