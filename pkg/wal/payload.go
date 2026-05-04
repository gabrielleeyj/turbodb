package wal

import (
	"encoding/binary"
	"fmt"
	"math"
	"sort"
)

// Typed payload encoders/decoders. The WAL itself is agnostic to payload
// shape — these helpers exist so that engine code does not have to redefine a
// serialization format per record type.

// InsertPayload describes a vector insertion.
type InsertPayload struct {
	Collection string
	ID         string
	Values     []float32
	Metadata   map[string]string
}

// DeletePayload describes a vector deletion.
type DeletePayload struct {
	Collection string
	ID         string
}

// SegmentSealedPayload describes a segment seal event.
type SegmentSealedPayload struct {
	Collection string
	SegmentID  string
	FilePath   string
	VectorCount uint64
}

// CheckpointPayload describes a durability checkpoint.
type CheckpointPayload struct {
	// LSN is the highest LSN whose effects are durable in the index/segments.
	LSN uint64
}

// EncodeInsert serializes an InsertPayload.
func EncodeInsert(p InsertPayload) ([]byte, error) {
	if p.Collection == "" {
		return nil, fmt.Errorf("wal: insert payload requires collection")
	}
	if p.ID == "" {
		return nil, fmt.Errorf("wal: insert payload requires id")
	}
	if len(p.Values) == 0 {
		return nil, fmt.Errorf("wal: insert payload requires non-empty values")
	}
	if len(p.Values) > math.MaxUint32 {
		return nil, fmt.Errorf("wal: insert payload values too large")
	}

	buf := make([]byte, 0, 64+len(p.Values)*4)
	buf = appendString(buf, p.Collection)
	buf = appendString(buf, p.ID)

	dim := uint32(len(p.Values))
	var dimBuf [4]byte
	binary.LittleEndian.PutUint32(dimBuf[:], dim)
	buf = append(buf, dimBuf[:]...)

	valBuf := make([]byte, 4)
	for _, v := range p.Values {
		binary.LittleEndian.PutUint32(valBuf, math.Float32bits(v))
		buf = append(buf, valBuf...)
	}

	buf = appendMetadata(buf, p.Metadata)
	return buf, nil
}

// DecodeInsert parses an InsertPayload from bytes.
func DecodeInsert(b []byte) (InsertPayload, error) {
	var p InsertPayload
	c, n, err := readStringAt(b, 0)
	if err != nil {
		return p, fmt.Errorf("wal: decode insert collection: %w", err)
	}
	p.Collection = c

	id, n2, err := readStringAt(b, n)
	if err != nil {
		return p, fmt.Errorf("wal: decode insert id: %w", err)
	}
	p.ID = id
	off := n2

	if off+4 > len(b) {
		return p, fmt.Errorf("wal: decode insert: short dim")
	}
	dim := int(binary.LittleEndian.Uint32(b[off : off+4]))
	off += 4
	if dim < 1 {
		return p, fmt.Errorf("wal: decode insert: invalid dim %d", dim)
	}
	if off+dim*4 > len(b) {
		return p, fmt.Errorf("wal: decode insert: short values")
	}
	p.Values = make([]float32, dim)
	for i := range dim {
		bits := binary.LittleEndian.Uint32(b[off : off+4])
		p.Values[i] = math.Float32frombits(bits)
		off += 4
	}

	meta, _, err := readMetadataAt(b, off)
	if err != nil {
		return p, fmt.Errorf("wal: decode insert metadata: %w", err)
	}
	p.Metadata = meta
	return p, nil
}

// EncodeDelete serializes a DeletePayload.
func EncodeDelete(p DeletePayload) ([]byte, error) {
	if p.Collection == "" {
		return nil, fmt.Errorf("wal: delete payload requires collection")
	}
	if p.ID == "" {
		return nil, fmt.Errorf("wal: delete payload requires id")
	}
	buf := make([]byte, 0, 16+len(p.Collection)+len(p.ID))
	buf = appendString(buf, p.Collection)
	buf = appendString(buf, p.ID)
	return buf, nil
}

// DecodeDelete parses a DeletePayload from bytes.
func DecodeDelete(b []byte) (DeletePayload, error) {
	var p DeletePayload
	c, n, err := readStringAt(b, 0)
	if err != nil {
		return p, fmt.Errorf("wal: decode delete collection: %w", err)
	}
	p.Collection = c

	id, _, err := readStringAt(b, n)
	if err != nil {
		return p, fmt.Errorf("wal: decode delete id: %w", err)
	}
	p.ID = id
	return p, nil
}

// EncodeSegmentSealed serializes a SegmentSealedPayload.
func EncodeSegmentSealed(p SegmentSealedPayload) ([]byte, error) {
	if p.Collection == "" || p.SegmentID == "" || p.FilePath == "" {
		return nil, fmt.Errorf("wal: segment_sealed payload requires collection, segment_id, and file_path")
	}
	buf := make([]byte, 0, 32+len(p.Collection)+len(p.SegmentID)+len(p.FilePath))
	buf = appendString(buf, p.Collection)
	buf = appendString(buf, p.SegmentID)
	buf = appendString(buf, p.FilePath)

	var countBuf [8]byte
	binary.LittleEndian.PutUint64(countBuf[:], p.VectorCount)
	buf = append(buf, countBuf[:]...)
	return buf, nil
}

// DecodeSegmentSealed parses a SegmentSealedPayload from bytes.
func DecodeSegmentSealed(b []byte) (SegmentSealedPayload, error) {
	var p SegmentSealedPayload
	c, n, err := readStringAt(b, 0)
	if err != nil {
		return p, fmt.Errorf("wal: decode segment_sealed collection: %w", err)
	}
	p.Collection = c

	seg, n2, err := readStringAt(b, n)
	if err != nil {
		return p, fmt.Errorf("wal: decode segment_sealed segment_id: %w", err)
	}
	p.SegmentID = seg

	fp, n3, err := readStringAt(b, n2)
	if err != nil {
		return p, fmt.Errorf("wal: decode segment_sealed file_path: %w", err)
	}
	p.FilePath = fp

	if n3+8 > len(b) {
		return p, fmt.Errorf("wal: decode segment_sealed: short vector_count")
	}
	p.VectorCount = binary.LittleEndian.Uint64(b[n3 : n3+8])
	return p, nil
}

// EncodeCheckpoint serializes a CheckpointPayload.
func EncodeCheckpoint(p CheckpointPayload) ([]byte, error) {
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], p.LSN)
	return buf[:], nil
}

// DecodeCheckpoint parses a CheckpointPayload from bytes.
func DecodeCheckpoint(b []byte) (CheckpointPayload, error) {
	if len(b) < 8 {
		return CheckpointPayload{}, fmt.Errorf("wal: decode checkpoint: short payload")
	}
	return CheckpointPayload{LSN: binary.LittleEndian.Uint64(b[:8])}, nil
}

// appendString appends a length-prefixed UTF-8 string.
func appendString(buf []byte, s string) []byte {
	var lenBuf [4]byte
	binary.LittleEndian.PutUint32(lenBuf[:], uint32(len(s)))
	buf = append(buf, lenBuf[:]...)
	buf = append(buf, s...)
	return buf
}

// readStringAt reads a length-prefixed string starting at offset off.
// Returns the string and the new offset.
func readStringAt(b []byte, off int) (string, int, error) {
	if off+4 > len(b) {
		return "", 0, fmt.Errorf("short length prefix")
	}
	length := int(binary.LittleEndian.Uint32(b[off : off+4]))
	off += 4
	if off+length > len(b) {
		return "", 0, fmt.Errorf("string body length %d exceeds buffer", length)
	}
	return string(b[off : off+length]), off + length, nil
}

// appendMetadata appends a uint16 count followed by length-prefixed key/value pairs.
// Keys are written in sorted order to make encoding deterministic.
func appendMetadata(buf []byte, meta map[string]string) []byte {
	var countBuf [2]byte
	binary.LittleEndian.PutUint16(countBuf[:], uint16(len(meta)))
	buf = append(buf, countBuf[:]...)

	if len(meta) == 0 {
		return buf
	}
	keys := make([]string, 0, len(meta))
	for k := range meta {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, k := range keys {
		buf = appendString(buf, k)
		buf = appendString(buf, meta[k])
	}
	return buf
}

// readMetadataAt reads a metadata map starting at offset off.
func readMetadataAt(b []byte, off int) (map[string]string, int, error) {
	if off+2 > len(b) {
		return nil, 0, fmt.Errorf("short metadata count")
	}
	count := int(binary.LittleEndian.Uint16(b[off : off+2]))
	off += 2
	if count == 0 {
		return nil, off, nil
	}

	meta := make(map[string]string, count)
	for i := range count {
		k, n, err := readStringAt(b, off)
		if err != nil {
			return nil, 0, fmt.Errorf("read metadata key %d: %w", i, err)
		}
		v, n2, err := readStringAt(b, n)
		if err != nil {
			return nil, 0, fmt.Errorf("read metadata value %d: %w", i, err)
		}
		meta[k] = v
		off = n2
	}
	return meta, off, nil
}
