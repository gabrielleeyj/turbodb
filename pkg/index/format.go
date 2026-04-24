package index

import (
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"io"
	"math"
	"os"
	"time"

	"github.com/gabrielleeyj/turbodb/pkg/codebook"
	"github.com/gabrielleeyj/turbodb/pkg/quantizer"
	"github.com/gabrielleeyj/turbodb/pkg/rotation"
)

// Segment file format:
//
//   ┌──────────────────────────────────┐
//   │ Header (64 bytes)                │
//   │   Magic      [4]byte "TQSG"     │
//   │   Version    uint32             │
//   │   Dim        uint32             │
//   │   BitWidth   uint32             │
//   │   RotSeed    uint64             │
//   │   CodebookDim uint32            │
//   │   CodebookBW  uint32            │
//   │   Count      uint64             │
//   │   CreatedAt  int64 (unix ns)    │
//   │   SealedAt   int64 (unix ns)    │
//   │   Reserved   [8]byte            │
//   ├──────────────────────────────────┤
//   │ Body                            │
//   │   IDs: length-prefixed strings  │
//   │   Norms: [Count]float32         │
//   │   Codes: per-vector packed bits │
//   ├──────────────────────────────────┤
//   │ Footer (8 bytes)                │
//   │   CRC32C  uint32 (over hdr+body)│
//   │   FooterMagic [4]byte "TQSF"   │
//   └──────────────────────────────────┘

const (
	segmentMagic  = "TQSG"
	footerMagic   = "TQSF"
	formatVersion = 1
	headerSize    = 64
	footerSize    = 8
)

// SegmentHeader is the fixed-size header at the start of a segment file.
type SegmentHeader struct {
	Magic       [4]byte
	Version     uint32
	Dim         uint32
	BitWidth    uint32
	RotSeed     uint64
	CodebookDim uint32
	CodebookBW  uint32
	Count       uint64
	CreatedAt   int64
	SealedAt    int64
	Reserved    [8]byte
}

// WriteSegmentFile writes a sealed segment to a file.
func WriteSegmentFile(path string, seg *SealedSegment) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("write segment file: %w", err)
	}
	defer f.Close()

	crc := crc32.New(crc32.MakeTable(crc32.Castagnoli))
	w := io.MultiWriter(f, crc)

	// Write header.
	hdr := SegmentHeader{
		Version:     formatVersion,
		Dim:         uint32(seg.dim),
		BitWidth:    uint32(seg.bitWidth),
		RotSeed:     seg.rotator.Seed(),
		CodebookDim: uint32(seg.cb.Dim()),
		CodebookBW:  uint32(seg.cb.BitWidth()),
		Count:       uint64(seg.count),
		CreatedAt:   seg.createdAt.UnixNano(),
		SealedAt:    seg.sealedAt.UnixNano(),
	}
	copy(hdr.Magic[:], segmentMagic)

	if err := binary.Write(w, binary.LittleEndian, &hdr); err != nil {
		return fmt.Errorf("write segment header: %w", err)
	}

	// Write IDs as length-prefixed strings.
	for _, id := range seg.ids {
		if err := writeString(w, id); err != nil {
			return fmt.Errorf("write segment ID: %w", err)
		}
	}

	// Write norms.
	if err := binary.Write(w, binary.LittleEndian, seg.norms); err != nil {
		return fmt.Errorf("write segment norms: %w", err)
	}

	// Write codes: for each vector, write the packed indices bytes.
	for i := range seg.codes {
		codeLen := uint32(len(seg.codes[i].Indices))
		if err := binary.Write(w, binary.LittleEndian, codeLen); err != nil {
			return fmt.Errorf("write code length %d: %w", i, err)
		}
		if _, err := w.Write(seg.codes[i].Indices); err != nil {
			return fmt.Errorf("write code data %d: %w", i, err)
		}
	}

	// Write footer: CRC32C + magic.
	checksum := crc.Sum32()
	if err := binary.Write(f, binary.LittleEndian, checksum); err != nil {
		return fmt.Errorf("write segment footer crc: %w", err)
	}
	if _, err := f.Write([]byte(footerMagic)); err != nil {
		return fmt.Errorf("write segment footer magic: %w", err)
	}

	return f.Close()
}

// ReadSegmentFile reads a sealed segment from a file. The caller provides the
// rotation and codebook which are identified by seed/dim/bw from the header.
func ReadSegmentFile(path string, segID string, rot rotation.Rotator, cb *codebook.Codebook) (*SealedSegment, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read segment file: %w", err)
	}

	if len(data) < headerSize+footerSize {
		return nil, fmt.Errorf("read segment file: file too small (%d bytes)", len(data))
	}

	// Verify footer magic.
	fmSlice := data[len(data)-4:]
	if string(fmSlice) != footerMagic {
		return nil, fmt.Errorf("read segment file: invalid footer magic")
	}

	// Verify CRC32C.
	storedCRC := binary.LittleEndian.Uint32(data[len(data)-8 : len(data)-4])
	computedCRC := crc32.Checksum(data[:len(data)-footerSize], crc32.MakeTable(crc32.Castagnoli))
	if storedCRC != computedCRC {
		return nil, fmt.Errorf("read segment file: CRC mismatch (stored=%x, computed=%x)", storedCRC, computedCRC)
	}

	// Parse header.
	var hdr SegmentHeader
	hdrBytes := data[:headerSize]
	copy(hdr.Magic[:], hdrBytes[0:4])
	if string(hdr.Magic[:]) != segmentMagic {
		return nil, fmt.Errorf("read segment file: invalid header magic")
	}

	hdr.Version = binary.LittleEndian.Uint32(hdrBytes[4:8])
	if hdr.Version != formatVersion {
		return nil, fmt.Errorf("read segment file: unsupported version %d", hdr.Version)
	}

	hdr.Dim = binary.LittleEndian.Uint32(hdrBytes[8:12])
	hdr.BitWidth = binary.LittleEndian.Uint32(hdrBytes[12:16])
	hdr.RotSeed = binary.LittleEndian.Uint64(hdrBytes[16:24])
	hdr.CodebookDim = binary.LittleEndian.Uint32(hdrBytes[24:28])
	hdr.CodebookBW = binary.LittleEndian.Uint32(hdrBytes[28:32])
	hdr.Count = binary.LittleEndian.Uint64(hdrBytes[32:40])
	hdr.CreatedAt = int64(binary.LittleEndian.Uint64(hdrBytes[40:48]))
	hdr.SealedAt = int64(binary.LittleEndian.Uint64(hdrBytes[48:56]))

	// Validate header against provided rotator/codebook.
	if rot.Seed() != hdr.RotSeed {
		return nil, fmt.Errorf("read segment file: rotator seed mismatch (header=%d, provided=%d)",
			hdr.RotSeed, rot.Seed())
	}

	count := int(hdr.Count)
	dim := int(hdr.Dim)
	bitWidth := int(hdr.BitWidth)

	// Parse body — starts after header, ends before footer.
	body := data[headerSize : len(data)-footerSize]
	offset := 0

	// Read IDs.
	ids := make([]string, count)
	for i := range count {
		s, n, err := readString(body[offset:])
		if err != nil {
			return nil, fmt.Errorf("read segment ID %d: %w", i, err)
		}
		ids[i] = s
		offset += n
	}

	// Read norms.
	normsSize := count * 4
	if offset+normsSize > len(body) {
		return nil, fmt.Errorf("read segment file: body too short for norms")
	}
	norms := make([]float32, count)
	for i := range count {
		bits := binary.LittleEndian.Uint32(body[offset : offset+4])
		norms[i] = float32FromBits(bits)
		offset += 4
	}

	// Read codes.
	codes := make([]quantizer.Code, count)
	for i := range count {
		if offset+4 > len(body) {
			return nil, fmt.Errorf("read segment file: body too short for code length %d", i)
		}
		codeLen := int(binary.LittleEndian.Uint32(body[offset : offset+4]))
		offset += 4
		if offset+codeLen > len(body) {
			return nil, fmt.Errorf("read segment file: body too short for code data %d", i)
		}
		indices := make([]byte, codeLen)
		copy(indices, body[offset:offset+codeLen])
		offset += codeLen
		codes[i] = quantizer.Code{
			Indices:  indices,
			Norm:     norms[i],
			BitWidth: bitWidth,
			Dim:      rot.OutDim(),
		}
	}

	createdAt := time.Unix(0, hdr.CreatedAt)
	sealedAt := time.Unix(0, hdr.SealedAt)

	return NewSealedSegmentFromData(segID, dim, bitWidth, codes, ids, norms, rot, cb, createdAt, sealedAt)
}

// writeString writes a length-prefixed string.
func writeString(w io.Writer, s string) error {
	length := uint32(len(s))
	if err := binary.Write(w, binary.LittleEndian, length); err != nil {
		return err
	}
	_, err := io.WriteString(w, s)
	return err
}

// readString reads a length-prefixed string from a byte slice.
// Returns the string and the number of bytes consumed.
func readString(data []byte) (string, int, error) {
	if len(data) < 4 {
		return "", 0, fmt.Errorf("readString: data too short for length prefix")
	}
	length := int(binary.LittleEndian.Uint32(data[:4]))
	if 4+length > len(data) {
		return "", 0, fmt.Errorf("readString: data too short for string of length %d", length)
	}
	return string(data[4 : 4+length]), 4 + length, nil
}

// float32FromBits converts uint32 bits to float32.
func float32FromBits(bits uint32) float32 {
	return math.Float32frombits(bits)
}
