package wal

import (
	"testing"
	"time"
)

func TestAppendBatchAssignsContiguousLSNs(t *testing.T) {
	dir := t.TempDir()
	w, err := Open(Config{Dir: dir})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = w.Close() }()

	payloads := [][]byte{[]byte("a"), []byte("b"), []byte("c")}
	first, err := w.AppendBatch(OpInsert, payloads)
	if err != nil {
		t.Fatal(err)
	}
	if w.NextLSN() != first+3 {
		t.Errorf("NextLSN: got %d, want %d", w.NextLSN(), first+3)
	}

	// Records are durable and readable in order.
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	var got []Record
	if err := Iterate(dir, IterateOptions{}, func(r Record) error {
		got = append(got, r)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 {
		t.Fatalf("records: got %d, want 3", len(got))
	}
	for i, r := range got {
		if r.LSN != first+uint64(i) { // #nosec G115 -- test index
			t.Errorf("record %d: LSN %d, want %d", i, r.LSN, first+uint64(i))
		}
		if string(r.Payload) != string(payloads[i]) {
			t.Errorf("record %d: payload %q, want %q", i, r.Payload, payloads[i])
		}
	}
}

func TestAppendBatchGroupCommitSharesOneWait(t *testing.T) {
	w, err := Open(Config{
		Dir:                 t.TempDir(),
		FsyncPolicy:         FsyncGroupCommit,
		GroupCommitInterval: 20 * time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = w.Close() }()

	// 100 records through AppendBatch must take ~one group interval, not 100.
	payloads := make([][]byte, 100)
	for i := range payloads {
		payloads[i] = []byte("payload")
	}
	start := time.Now()
	if _, err := w.AppendBatch(OpInsert, payloads); err != nil {
		t.Fatal(err)
	}
	elapsed := time.Since(start)
	// Generous bound: a per-record wait would take >= 100 * 20ms = 2s.
	if elapsed > 500*time.Millisecond {
		t.Errorf("AppendBatch took %s; group commit is not batching", elapsed)
	}
}

func TestAppendBatchValidation(t *testing.T) {
	w, err := Open(Config{Dir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.AppendBatch(OpInsert, nil); err == nil {
		t.Error("expected error for empty batch")
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := w.AppendBatch(OpInsert, [][]byte{[]byte("x")}); err == nil {
		t.Error("expected error on closed WAL")
	}
}
