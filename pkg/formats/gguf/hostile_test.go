package gguf

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// hostileHeader hand-builds a GGUF v3 header with the given tensor and KV
// counts followed by extra bytes.
func hostileHeader(nTensors, nKV uint64, rest []byte) []byte {
	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.LittleEndian, Magic)
	_ = binary.Write(&buf, binary.LittleEndian, Version)
	_ = binary.Write(&buf, binary.LittleEndian, nTensors)
	_ = binary.Write(&buf, binary.LittleEndian, nKV)
	buf.Write(rest)
	return buf.Bytes()
}

// TestReaderRejectsHostileArrayCount ensures a metadata array declaring an
// enormous element count fails with an error instead of attempting the
// allocation up front.
func TestReaderRejectsHostileArrayCount(t *testing.T) {
	// Arrange: one KV whose value is an array claiming 2^62 elements.
	var kv bytes.Buffer
	_ = binary.Write(&kv, binary.LittleEndian, uint64(1)) // key length
	kv.WriteByte('k')
	_ = binary.Write(&kv, binary.LittleEndian, uint32(mvArray))
	_ = binary.Write(&kv, binary.LittleEndian, uint32(mvUint8)) // element type
	_ = binary.Write(&kv, binary.LittleEndian, uint64(1)<<62)   // element count
	data := hostileHeader(0, 1, kv.Bytes())

	// Act
	_, err := NewReader(bytes.NewReader(data), int64(len(data)))

	// Assert
	if err == nil {
		t.Fatal("expected error for implausible array element count")
	}
}

// TestReaderRejectsHostileHeaderCounts ensures tensor/KV counts wildly
// exceeding what the file could contain are rejected before the maps are
// size-hinted with them.
func TestReaderRejectsHostileHeaderCounts(t *testing.T) {
	tests := []struct {
		name     string
		nTensors uint64
		nKV      uint64
	}{
		{name: "huge kv count", nKV: 1 << 60},
		{name: "huge tensor count", nTensors: 1 << 60},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data := hostileHeader(tt.nTensors, tt.nKV, nil)

			_, err := NewReader(bytes.NewReader(data), int64(len(data)))

			if err == nil {
				t.Fatal("expected error for implausible header counts")
			}
		})
	}
}
