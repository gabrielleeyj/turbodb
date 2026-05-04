package wal

import (
	"reflect"
	"testing"
)

func TestEncodeDecodeInsert(t *testing.T) {
	t.Parallel()
	original := InsertPayload{
		Collection: "vectors",
		ID:         "doc-42",
		Values:     []float32{0.1, -0.2, 0.3, 0.4},
		Metadata:   map[string]string{"source": "test", "lang": "en"},
	}

	buf, err := EncodeInsert(original)
	if err != nil {
		t.Fatalf("EncodeInsert: %v", err)
	}

	got, err := DecodeInsert(buf)
	if err != nil {
		t.Fatalf("DecodeInsert: %v", err)
	}

	if !reflect.DeepEqual(got, original) {
		t.Fatalf("round-trip mismatch:\n  got:  %+v\n  want: %+v", got, original)
	}
}

func TestEncodeInsertValidation(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name string
		p    InsertPayload
	}{
		{"missing collection", InsertPayload{ID: "x", Values: []float32{1}}},
		{"missing id", InsertPayload{Collection: "c", Values: []float32{1}}},
		{"empty values", InsertPayload{Collection: "c", ID: "x"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := EncodeInsert(tc.p); err == nil {
				t.Fatalf("expected error for %q, got nil", tc.name)
			}
		})
	}
}

func TestEncodeDecodeDelete(t *testing.T) {
	t.Parallel()
	p := DeletePayload{Collection: "vectors", ID: "doc-7"}
	buf, err := EncodeDelete(p)
	if err != nil {
		t.Fatal(err)
	}
	got, err := DecodeDelete(buf)
	if err != nil {
		t.Fatal(err)
	}
	if got != p {
		t.Fatalf("round-trip mismatch: got %+v, want %+v", got, p)
	}
}

func TestEncodeDecodeSegmentSealed(t *testing.T) {
	t.Parallel()
	p := SegmentSealedPayload{
		Collection:  "vectors",
		SegmentID:   "vectors-seg-0001",
		FilePath:    "/var/turbodb/segments/vectors-seg-0001.tqsg",
		VectorCount: 1_000_000,
	}
	buf, err := EncodeSegmentSealed(p)
	if err != nil {
		t.Fatal(err)
	}
	got, err := DecodeSegmentSealed(buf)
	if err != nil {
		t.Fatal(err)
	}
	if got != p {
		t.Fatalf("round-trip mismatch: got %+v, want %+v", got, p)
	}
}

func TestEncodeDecodeCheckpoint(t *testing.T) {
	t.Parallel()
	p := CheckpointPayload{LSN: 9_999_999}
	buf, err := EncodeCheckpoint(p)
	if err != nil {
		t.Fatal(err)
	}
	got, err := DecodeCheckpoint(buf)
	if err != nil {
		t.Fatal(err)
	}
	if got != p {
		t.Fatalf("round-trip mismatch: got %+v, want %+v", got, p)
	}
}

func TestInsertEmptyMetadata(t *testing.T) {
	t.Parallel()
	p := InsertPayload{
		Collection: "c",
		ID:         "x",
		Values:     []float32{1, 2, 3},
	}
	buf, err := EncodeInsert(p)
	if err != nil {
		t.Fatal(err)
	}
	got, err := DecodeInsert(buf)
	if err != nil {
		t.Fatal(err)
	}
	if got.Metadata != nil {
		t.Fatalf("expected nil metadata, got %v", got.Metadata)
	}
}
