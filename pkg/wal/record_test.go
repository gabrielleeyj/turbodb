package wal

import (
	"bytes"
	"errors"
	"io"
	"testing"
)

func TestRecordRoundTrip(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		rec  Record
	}{
		{"insert", Record{LSN: 1, Type: OpInsert, Payload: []byte("hello")}},
		{"delete", Record{LSN: 2, Type: OpDelete, Payload: []byte("id-42")}},
		{"empty payload", Record{LSN: 99, Type: OpCheckpoint}},
		{"large payload", Record{LSN: 1234, Type: OpInsert, Payload: bytes.Repeat([]byte{0xAB}, 4096)}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			var buf bytes.Buffer
			n, err := writeRecord(&buf, tc.rec)
			if err != nil {
				t.Fatalf("writeRecord: %v", err)
			}
			if n != EncodedSize(len(tc.rec.Payload)) {
				t.Fatalf("size mismatch: got %d, want %d", n, EncodedSize(len(tc.rec.Payload)))
			}

			got, _, err := readRecord(&buf)
			if err != nil {
				t.Fatalf("readRecord: %v", err)
			}
			if got.LSN != tc.rec.LSN {
				t.Errorf("LSN: got %d, want %d", got.LSN, tc.rec.LSN)
			}
			if got.Type != tc.rec.Type {
				t.Errorf("Type: got %v, want %v", got.Type, tc.rec.Type)
			}
			if !bytes.Equal(got.Payload, tc.rec.Payload) {
				t.Errorf("Payload mismatch")
			}
		})
	}
}

func TestRecordCorruption(t *testing.T) {
	t.Parallel()

	t.Run("crc mismatch", func(t *testing.T) {
		var buf bytes.Buffer
		_, err := writeRecord(&buf, Record{LSN: 1, Type: OpInsert, Payload: []byte("data")})
		if err != nil {
			t.Fatal(err)
		}
		// Flip a payload byte.
		raw := buf.Bytes()
		raw[len(raw)-5] ^= 0xFF
		_, _, err = readRecord(bytes.NewReader(raw))
		if !errors.Is(err, ErrCorruptRecord) {
			t.Fatalf("expected ErrCorruptRecord, got %v", err)
		}
	})

	t.Run("truncated body", func(t *testing.T) {
		var buf bytes.Buffer
		_, err := writeRecord(&buf, Record{LSN: 1, Type: OpInsert, Payload: []byte("data")})
		if err != nil {
			t.Fatal(err)
		}
		raw := buf.Bytes()[:8] // only header + a few bytes
		_, _, err = readRecord(bytes.NewReader(raw))
		if !errors.Is(err, ErrCorruptRecord) {
			t.Fatalf("expected ErrCorruptRecord, got %v", err)
		}
	})

	t.Run("eof at boundary", func(t *testing.T) {
		_, _, err := readRecord(bytes.NewReader(nil))
		if !errors.Is(err, io.EOF) {
			t.Fatalf("expected io.EOF, got %v", err)
		}
	})
}

func TestRecordTypeString(t *testing.T) {
	t.Parallel()
	tests := map[RecordType]string{
		OpInsert:        "insert",
		OpDelete:        "delete",
		OpSegmentSealed: "segment_sealed",
		OpCheckpoint:    "checkpoint",
	}
	for typ, want := range tests {
		if got := typ.String(); got != want {
			t.Errorf("RecordType(%d).String() = %q, want %q", typ, got, want)
		}
	}
}
