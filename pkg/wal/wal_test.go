package wal

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

func mustOpen(t *testing.T, dir string, opts ...func(*Config)) *WAL {
	t.Helper()
	cfg := Config{Dir: dir}
	for _, o := range opts {
		o(&cfg)
	}
	w, err := Open(cfg)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { _ = w.Close() })
	return w
}

func TestAppendIterateRoundTrip(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	w := mustOpen(t, dir)

	payloads := []InsertPayload{
		{Collection: "c", ID: "a", Values: []float32{1, 2}},
		{Collection: "c", ID: "b", Values: []float32{3, 4}},
		{Collection: "c", ID: "c", Values: []float32{5, 6}},
	}

	var lsns []uint64
	for _, p := range payloads {
		buf, err := EncodeInsert(p)
		if err != nil {
			t.Fatal(err)
		}
		lsn, err := w.Append(OpInsert, buf)
		if err != nil {
			t.Fatal(err)
		}
		lsns = append(lsns, lsn)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	var seen []Record
	if err := Iterate(dir, IterateOptions{}, func(r Record) error {
		seen = append(seen, r)
		return nil
	}); err != nil {
		t.Fatalf("Iterate: %v", err)
	}

	if len(seen) != len(payloads) {
		t.Fatalf("expected %d records, got %d", len(payloads), len(seen))
	}
	for i, r := range seen {
		if r.LSN != lsns[i] {
			t.Errorf("record %d: lsn = %d, want %d", i, r.LSN, lsns[i])
		}
		if r.Type != OpInsert {
			t.Errorf("record %d: type = %v, want OpInsert", i, r.Type)
		}
		decoded, err := DecodeInsert(r.Payload)
		if err != nil {
			t.Errorf("record %d: decode: %v", i, err)
			continue
		}
		if decoded.ID != payloads[i].ID {
			t.Errorf("record %d: id = %q, want %q", i, decoded.ID, payloads[i].ID)
		}
	}
}

func TestLSNMonotonicAndRecovery(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()

	w1 := mustOpen(t, dir)
	for i := range 5 {
		_, err := w1.Append(OpInsert, []byte(fmt.Sprintf("v%d", i)))
		if err != nil {
			t.Fatal(err)
		}
	}
	if err := w1.Close(); err != nil {
		t.Fatal(err)
	}

	// Reopen; nextLSN should resume after the last record.
	w2 := mustOpen(t, dir)
	defer w2.Close()
	if got := w2.NextLSN(); got != 6 {
		t.Errorf("NextLSN after reopen = %d, want 6", got)
	}
	lsn, err := w2.Append(OpInsert, []byte("v5"))
	if err != nil {
		t.Fatal(err)
	}
	if lsn != 6 {
		t.Errorf("Append after reopen: lsn = %d, want 6", lsn)
	}
}

func TestFileRotation(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	w := mustOpen(t, dir, func(c *Config) {
		c.MaxFileBytes = 256 // tiny — force rotation
	})

	for i := range 50 {
		payload := []byte(fmt.Sprintf("record-%03d-padding-padding-padding", i))
		if _, err := w.Append(OpInsert, payload); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Sync(); err != nil {
		t.Fatal(err)
	}

	files, err := w.Files()
	if err != nil {
		t.Fatal(err)
	}
	if len(files) < 2 {
		t.Fatalf("expected file rotation, got %d files", len(files))
	}

	// All records should still be readable in order.
	var count int
	var prevLSN uint64
	if err := Iterate(dir, IterateOptions{}, func(r Record) error {
		if r.LSN <= prevLSN && prevLSN != 0 {
			t.Errorf("LSN went backwards: %d after %d", r.LSN, prevLSN)
		}
		prevLSN = r.LSN
		count++
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if count != 50 {
		t.Errorf("read %d records after rotation, want 50", count)
	}
}

func TestIterateFromLSN(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	w := mustOpen(t, dir)
	for i := range 10 {
		_, err := w.Append(OpInsert, []byte{byte(i)})
		if err != nil {
			t.Fatal(err)
		}
	}
	w.Close()

	var seen []uint64
	if err := Iterate(dir, IterateOptions{FromLSN: 5}, func(r Record) error {
		seen = append(seen, r.LSN)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	want := []uint64{5, 6, 7, 8, 9, 10}
	if len(seen) != len(want) {
		t.Fatalf("got %v, want %v", seen, want)
	}
	for i, v := range want {
		if seen[i] != v {
			t.Errorf("seen[%d] = %d, want %d", i, seen[i], v)
		}
	}
}

func TestStopIteration(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	w := mustOpen(t, dir)
	for i := range 10 {
		_, err := w.Append(OpInsert, []byte{byte(i)})
		if err != nil {
			t.Fatal(err)
		}
	}
	w.Close()

	var count int
	err := Iterate(dir, IterateOptions{}, func(r Record) error {
		count++
		if count == 3 {
			return ErrStopIteration
		}
		return nil
	})
	if err != nil {
		t.Fatalf("Iterate returned %v, want nil", err)
	}
	if count != 3 {
		t.Errorf("count = %d, want 3", count)
	}
}

func TestCheckpointAndTruncate(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	w := mustOpen(t, dir, func(c *Config) {
		c.MaxFileBytes = 200 // force multiple files
	})

	var lastInsertLSN uint64
	for i := range 20 {
		lsn, err := w.Append(OpInsert, []byte(fmt.Sprintf("rec-%02d", i)))
		if err != nil {
			t.Fatal(err)
		}
		lastInsertLSN = lsn
	}

	// Write checkpoint.
	cpBuf, err := EncodeCheckpoint(CheckpointPayload{LSN: lastInsertLSN})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.Append(OpCheckpoint, cpBuf); err != nil {
		t.Fatal(err)
	}
	if err := w.Sync(); err != nil {
		t.Fatal(err)
	}

	// Find checkpoint LSN before truncate.
	cp, err := LastCheckpointLSN(dir)
	if err != nil {
		t.Fatal(err)
	}
	if cp != lastInsertLSN {
		t.Errorf("LastCheckpointLSN = %d, want %d", cp, lastInsertLSN)
	}

	filesBefore, err := w.Files()
	if err != nil {
		t.Fatal(err)
	}

	// Truncate everything strictly before lastInsertLSN.
	if err := w.Truncate(lastInsertLSN); err != nil {
		t.Fatal(err)
	}

	filesAfter, err := w.Files()
	if err != nil {
		t.Fatal(err)
	}
	if len(filesAfter) >= len(filesBefore) {
		t.Errorf("expected fewer files after truncate: before=%d after=%d", len(filesBefore), len(filesAfter))
	}
	w.Close()
}

func TestGroupCommit(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	w := mustOpen(t, dir, func(c *Config) {
		c.FsyncPolicy = FsyncGroupCommit
		c.GroupCommitInterval = 5 * time.Millisecond
	})

	var wg sync.WaitGroup
	const writers = 8
	const perWriter = 50
	errs := make([]error, writers)

	for i := range writers {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			for j := range perWriter {
				if _, err := w.Append(OpInsert, []byte(fmt.Sprintf("w%d-r%d", idx, j))); err != nil {
					errs[idx] = err
					return
				}
			}
		}(i)
	}
	wg.Wait()

	for i, e := range errs {
		if e != nil {
			t.Fatalf("writer %d: %v", i, e)
		}
	}
	w.Close()

	var count int
	if err := Iterate(dir, IterateOptions{}, func(r Record) error { count++; return nil }); err != nil {
		t.Fatal(err)
	}
	if count != writers*perWriter {
		t.Errorf("count = %d, want %d", count, writers*perWriter)
	}
}

func TestRecoveryAfterPartialWrite(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	w := mustOpen(t, dir)
	for i := range 5 {
		_, err := w.Append(OpInsert, []byte{byte(i)})
		if err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	// Append garbage bytes to simulate a torn write.
	files, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(files) == 0 {
		t.Fatal("no WAL files written")
	}
	path := filepath.Join(dir, files[0].Name())
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Write([]byte{0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x01}); err != nil {
		t.Fatal(err)
	}
	f.Close()

	// Recovery should read the 5 valid records and stop at the corruption.
	var count int
	if err := Iterate(dir, IterateOptions{}, func(r Record) error { count++; return nil }); err != nil {
		t.Fatalf("Iterate: %v", err)
	}
	if count != 5 {
		t.Errorf("recovered %d records, want 5", count)
	}

	// With StopOnCorruption, expect an error.
	err = Iterate(dir, IterateOptions{StopOnCorruption: true}, func(r Record) error { return nil })
	if err == nil || !errors.Is(err, ErrCorruptRecord) {
		t.Errorf("expected ErrCorruptRecord, got %v", err)
	}
}

func TestAppendAfterClose(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	w, err := Open(Config{Dir: dir})
	if err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := w.Append(OpInsert, []byte("x")); err == nil {
		t.Error("expected error appending after close")
	}
}

func TestEmptyDirOpen(t *testing.T) {
	t.Parallel()
	dir := filepath.Join(t.TempDir(), "fresh")
	w, err := Open(Config{Dir: dir})
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	if got := w.NextLSN(); got != 1 {
		t.Errorf("NextLSN on empty WAL = %d, want 1", got)
	}
}
