package replication

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFileCheckpointRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sync.ckpt")
	cp, err := NewFileCheckpoint(path)
	if err != nil {
		t.Fatal(err)
	}

	// No file yet: fresh start.
	lsn, err := cp.Load()
	if err != nil || lsn != 0 {
		t.Fatalf("initial Load: lsn=%d err=%v, want 0, nil", lsn, err)
	}

	if err := cp.Save(12345); err != nil {
		t.Fatalf("Save: %v", err)
	}
	lsn, err = cp.Load()
	if err != nil || lsn != 12345 {
		t.Fatalf("Load: lsn=%d err=%v, want 12345, nil", lsn, err)
	}

	// Overwrite advances.
	if err := cp.Save(99999); err != nil {
		t.Fatalf("Save: %v", err)
	}
	lsn, _ = cp.Load()
	if lsn != 99999 {
		t.Errorf("Load after overwrite: got %d, want 99999", lsn)
	}
}

func TestFileCheckpointCorruption(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sync.ckpt")
	cp, err := NewFileCheckpoint(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := cp.Save(42); err != nil {
		t.Fatal(err)
	}

	// Flip a byte in the LSN: CRC must catch it.
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	data[0] ^= 0xFF
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := cp.Load(); err == nil || !strings.Contains(err.Error(), "crc mismatch") {
		t.Errorf("Load on corrupt file: got %v, want crc mismatch", err)
	}

	// Truncated file is rejected.
	if err := os.WriteFile(path, data[:5], 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := cp.Load(); err == nil || !strings.Contains(err.Error(), "unexpected size") {
		t.Errorf("Load on truncated file: got %v, want unexpected size", err)
	}
}

func TestNewFileCheckpointValidation(t *testing.T) {
	if _, err := NewFileCheckpoint(""); err == nil {
		t.Error("expected error for empty path")
	}
	if _, err := NewFileCheckpoint(filepath.Join(t.TempDir(), "nope", "sync.ckpt")); err == nil {
		t.Error("expected error for missing parent dir")
	}
}
