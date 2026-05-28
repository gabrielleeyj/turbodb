// Package pgproto defines the binary IPC protocol spoken between the
// pg_turboquant PostgreSQL extension (C) and the TurboDB engine daemon (Go)
// over a SOCK_STREAM Unix socket.
//
// Wire frame:
//
//	[4 bytes big-endian uint32 length L]   // bytes that follow (opcode + payload)
//	[2 bytes big-endian uint16 opcode]
//	[L-2 bytes payload]                    // little-endian, schema-versioned
//
// The length prefix is big-endian (network order) so the C side can use htonl;
// payload bodies are little-endian to match x86_64/ARM64 host layout and avoid
// per-field byte swapping in the hot path.
package pgproto

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
)

// Opcode identifies a message kind.
type Opcode uint16

const (
	OpBuildBegin  Opcode = 1
	OpBuildVector Opcode = 2
	OpBuildCommit Opcode = 3
	OpInsert      Opcode = 4
	OpDelete      Opcode = 5
	OpSearchBegin Opcode = 6
	OpSearchNext  Opcode = 7
	OpSearchEnd   Opcode = 8
	OpStats       Opcode = 9
	OpShutdown    Opcode = 10

	// OpAck and OpError are engine→client replies.
	OpAck   Opcode = 100
	OpError Opcode = 101
	// OpResult carries a single search result row in reply to OpSearchNext.
	OpResult Opcode = 102
)

// SchemaVersion is the payload schema version, prefixed to every payload body.
const SchemaVersion uint16 = 1

// MaxFrameBytes bounds a single frame to guard against corrupt length prefixes.
const MaxFrameBytes = 64 << 20 // 64 MiB

// ErrFrameTooLarge is returned when a frame exceeds MaxFrameBytes.
var ErrFrameTooLarge = errors.New("pgproto: frame exceeds maximum size")

// Frame is a decoded protocol message.
type Frame struct {
	Opcode  Opcode
	Payload []byte // payload body WITHOUT the schema-version prefix
}

// WriteFrame encodes opcode + payload (prepending the schema version) and
// writes a single length-prefixed frame to w.
func WriteFrame(w io.Writer, op Opcode, payload []byte) error {
	bodyLen := 2 + 2 + len(payload) // opcode + schemaVersion + payload
	if bodyLen > MaxFrameBytes {
		return ErrFrameTooLarge
	}
	buf := make([]byte, 4+bodyLen)
	binary.BigEndian.PutUint32(buf[0:], uint32(bodyLen))
	binary.BigEndian.PutUint16(buf[4:], uint16(op))
	binary.LittleEndian.PutUint16(buf[6:], SchemaVersion)
	copy(buf[8:], payload)
	if _, err := w.Write(buf); err != nil {
		return fmt.Errorf("pgproto: write frame: %w", err)
	}
	return nil
}

// ReadFrame reads one length-prefixed frame, validating the schema version.
func ReadFrame(r io.Reader) (Frame, error) {
	var lenBuf [4]byte
	if _, err := io.ReadFull(r, lenBuf[:]); err != nil {
		return Frame{}, err // io.EOF propagates for clean connection close
	}
	bodyLen := binary.BigEndian.Uint32(lenBuf[:])
	if bodyLen < 4 {
		return Frame{}, fmt.Errorf("pgproto: frame body too short (%d bytes)", bodyLen)
	}
	if bodyLen > MaxFrameBytes {
		return Frame{}, ErrFrameTooLarge
	}
	body := make([]byte, bodyLen)
	if _, err := io.ReadFull(r, body); err != nil {
		return Frame{}, fmt.Errorf("pgproto: read frame body: %w", err)
	}
	op := Opcode(binary.BigEndian.Uint16(body[0:]))
	version := binary.LittleEndian.Uint16(body[2:])
	if version != SchemaVersion {
		return Frame{}, fmt.Errorf("pgproto: unsupported schema version %d (want %d)", version, SchemaVersion)
	}
	return Frame{Opcode: op, Payload: body[4:]}, nil
}

// String renders an opcode name for logging.
func (o Opcode) String() string {
	switch o {
	case OpBuildBegin:
		return "BUILD_BEGIN"
	case OpBuildVector:
		return "BUILD_VECTOR"
	case OpBuildCommit:
		return "BUILD_COMMIT"
	case OpInsert:
		return "INSERT"
	case OpDelete:
		return "DELETE"
	case OpSearchBegin:
		return "SEARCH_BEGIN"
	case OpSearchNext:
		return "SEARCH_NEXT"
	case OpSearchEnd:
		return "SEARCH_END"
	case OpStats:
		return "STATS"
	case OpShutdown:
		return "SHUTDOWN"
	case OpAck:
		return "ACK"
	case OpError:
		return "ERROR"
	case OpResult:
		return "RESULT"
	default:
		return fmt.Sprintf("OP(%d)", uint16(o))
	}
}
