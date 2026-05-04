// Package wal implements a write-ahead log for the TurboDB engine. The WAL
// provides durability for inserts, deletes, segment seals, and checkpoints.
// Records are appended to per-file segments rotated at a configurable size.
//
// Recovery replays records from the most recent checkpoint LSN forward; older
// files can be truncated once a checkpoint is durable.
package wal

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
)

// RecordType identifies the operation encoded in a WAL record.
type RecordType uint8

const (
	// OpInsert records a vector insertion.
	OpInsert RecordType = 1
	// OpDelete records a vector deletion (tombstone).
	OpDelete RecordType = 2
	// OpSegmentSealed records that a growing segment was sealed to disk.
	OpSegmentSealed RecordType = 3
	// OpCheckpoint records a durability watermark; records at or below the
	// checkpoint LSN have been applied to durable state.
	OpCheckpoint RecordType = 4
)

// String returns a human-readable record type name.
func (t RecordType) String() string {
	switch t {
	case OpInsert:
		return "insert"
	case OpDelete:
		return "delete"
	case OpSegmentSealed:
		return "segment_sealed"
	case OpCheckpoint:
		return "checkpoint"
	default:
		return fmt.Sprintf("unknown(%d)", uint8(t))
	}
}

// Record is a single WAL entry. Payload encoding is defined by the record type
// and is the responsibility of the caller (see Encode*/Decode* helpers).
type Record struct {
	LSN     uint64
	Type    RecordType
	Payload []byte
}

// Frame layout on disk (little-endian):
//
//   [bodyLen: uint32][LSN: uint64][type: uint8][payload: bodyLen-9 bytes][crc32c: uint32]
//
// bodyLen covers LSN+type+payload (the bytes the CRC is computed over).
// The CRC is appended after the body; total record size on disk is
// 4 + bodyLen + 4.

const (
	frameHeaderSize = 4 // bodyLen prefix
	frameTrailerSize = 4 // crc32c suffix
	bodyMinSize     = 8 + 1 // LSN + type
)

// crcTable is the Castagnoli polynomial used throughout the project.
var crcTable = crc32.MakeTable(crc32.Castagnoli)

// ErrCorruptRecord is returned when a record fails CRC validation or its
// framing is malformed. Callers iterating the WAL during recovery should treat
// this as a tail truncation point (records past the corruption are unreadable).
var ErrCorruptRecord = errors.New("wal: corrupt record")

// EncodedSize returns the total on-disk size of a record with the given payload length.
func EncodedSize(payloadLen int) int {
	return frameHeaderSize + bodyMinSize + payloadLen + frameTrailerSize
}

// writeRecord serializes a record to w. Returns the number of bytes written.
func writeRecord(w io.Writer, rec Record) (int, error) {
	bodyLen := uint32(bodyMinSize + len(rec.Payload))

	buf := make([]byte, frameHeaderSize+int(bodyLen)+frameTrailerSize)
	binary.LittleEndian.PutUint32(buf[0:4], bodyLen)
	binary.LittleEndian.PutUint64(buf[4:12], rec.LSN)
	buf[12] = byte(rec.Type)
	copy(buf[13:13+len(rec.Payload)], rec.Payload)

	bodyStart := frameHeaderSize
	bodyEnd := bodyStart + int(bodyLen)
	crc := crc32.Checksum(buf[bodyStart:bodyEnd], crcTable)
	binary.LittleEndian.PutUint32(buf[bodyEnd:bodyEnd+4], crc)

	n, err := w.Write(buf)
	if err != nil {
		return n, fmt.Errorf("wal: write record: %w", err)
	}
	return n, nil
}

// readRecord deserializes a single record from r. Returns io.EOF if r is at end
// of stream. Returns ErrCorruptRecord (wrapped) if the frame is malformed or
// the CRC does not match.
func readRecord(r io.Reader) (Record, int, error) {
	var hdr [frameHeaderSize]byte
	if _, err := io.ReadFull(r, hdr[:]); err != nil {
		if errors.Is(err, io.EOF) {
			return Record{}, 0, io.EOF
		}
		if errors.Is(err, io.ErrUnexpectedEOF) {
			return Record{}, 0, fmt.Errorf("%w: short header", ErrCorruptRecord)
		}
		return Record{}, 0, fmt.Errorf("wal: read frame header: %w", err)
	}

	bodyLen := binary.LittleEndian.Uint32(hdr[:])
	if bodyLen < bodyMinSize {
		return Record{}, frameHeaderSize, fmt.Errorf("%w: body length %d too small", ErrCorruptRecord, bodyLen)
	}

	body := make([]byte, bodyLen)
	if _, err := io.ReadFull(r, body); err != nil {
		return Record{}, frameHeaderSize, fmt.Errorf("%w: short body: %w", ErrCorruptRecord, err)
	}

	var trailer [frameTrailerSize]byte
	if _, err := io.ReadFull(r, trailer[:]); err != nil {
		return Record{}, frameHeaderSize + int(bodyLen), fmt.Errorf("%w: short trailer: %w", ErrCorruptRecord, err)
	}

	storedCRC := binary.LittleEndian.Uint32(trailer[:])
	computedCRC := crc32.Checksum(body, crcTable)
	if storedCRC != computedCRC {
		return Record{}, frameHeaderSize + int(bodyLen) + frameTrailerSize,
			fmt.Errorf("%w: crc mismatch (stored=%x computed=%x)", ErrCorruptRecord, storedCRC, computedCRC)
	}

	rec := Record{
		LSN:  binary.LittleEndian.Uint64(body[0:8]),
		Type: RecordType(body[8]),
	}
	if len(body) > bodyMinSize {
		rec.Payload = make([]byte, len(body)-bodyMinSize)
		copy(rec.Payload, body[bodyMinSize:])
	}

	return rec, frameHeaderSize + int(bodyLen) + frameTrailerSize, nil
}
