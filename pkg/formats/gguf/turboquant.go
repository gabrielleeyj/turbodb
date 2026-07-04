package gguf

import (
	"encoding/binary"
	"fmt"
)

// TurboQuant GGUF block layout (see docs/gguf-turboquant.md):
//
//	per block of 32 coordinates:
//	  [2 bytes] float16 norm
//	  [2 bytes] uint16  seed offset into the per-model rotator table
//	  [ceil(32*b/8) bytes] packed b-bit codes (little-endian bit order)
//
// The norm + seed header is 4 bytes; b is the bit width (1..8).
const (
	TQBlockElems  = 32
	tqHeaderBytes = 4
)

// TQBlockBytes returns the byte size of one TurboQuant block at bit width b.
func TQBlockBytes(b int) int {
	return tqHeaderBytes + (TQBlockElems*b+7)/8
}

// EncodeTurboQuantBlock packs a single 32-coordinate block. codes holds the
// per-coordinate b-bit code values (0..2^b-1); seedOffset indexes the rotator
// table. The result is exactly TQBlockBytes(b) bytes.
func EncodeTurboQuantBlock(norm float32, seedOffset uint16, codes []uint8, b int) ([]byte, error) {
	if b < 1 || b > 8 {
		return nil, fmt.Errorf("gguf: turboquant bit width %d out of range [1,8]", b)
	}
	if len(codes) != TQBlockElems {
		return nil, fmt.Errorf("gguf: turboquant block needs %d codes, got %d", TQBlockElems, len(codes))
	}
	out := make([]byte, TQBlockBytes(b))
	binary.LittleEndian.PutUint16(out[0:], float32ToHalf(norm))
	binary.LittleEndian.PutUint16(out[2:], seedOffset)

	body := out[tqHeaderBytes:]
	mask := uint8((1 << b) - 1)
	var bitPos int
	for _, code := range codes {
		v := uint32(code & mask)
		byteIdx := bitPos >> 3
		bitOff := bitPos & 7
		// Write up to b bits across at most two bytes (b<=8).
		body[byteIdx] |= byte(v << bitOff) // #nosec G115 -- deliberate bit packing
		if bitOff+b > 8 {
			body[byteIdx+1] |= byte(v >> (8 - bitOff)) // #nosec G115 -- deliberate bit packing
		}
		bitPos += b
	}
	return out, nil
}

// DecodeTurboQuantBlock unpacks a block produced by EncodeTurboQuantBlock.
func DecodeTurboQuantBlock(block []byte, b int) (norm float32, seedOffset uint16, codes []uint8, err error) {
	if b < 1 || b > 8 {
		return 0, 0, nil, fmt.Errorf("gguf: turboquant bit width %d out of range [1,8]", b)
	}
	if len(block) != TQBlockBytes(b) {
		return 0, 0, nil, fmt.Errorf("gguf: turboquant block expects %d bytes, got %d", TQBlockBytes(b), len(block))
	}
	norm = f16ToF32(binary.LittleEndian.Uint16(block[0:]))
	seedOffset = binary.LittleEndian.Uint16(block[2:])
	body := block[tqHeaderBytes:]
	codes = make([]uint8, TQBlockElems)
	mask := uint16((1 << b) - 1)
	var bitPos int
	for i := range codes {
		byteIdx := bitPos >> 3
		bitOff := bitPos & 7
		v := uint16(body[byteIdx]) >> bitOff
		if bitOff+b > 8 {
			v |= uint16(body[byteIdx+1]) << (8 - bitOff)
		}
		codes[i] = uint8(v & mask) // #nosec G115 -- masked to bit width
		bitPos += b
	}
	return norm, seedOffset, codes, nil
}
