package quantizer

import "fmt"

// PackBits packs indices (each in range [0, 2^bitWidth-1]) into a byte slice.
// Returns exactly ceil(bitWidth * len(indices) / 8) bytes.
func PackBits(indices []int, bitWidth int) ([]byte, error) {
	if bitWidth < 1 || bitWidth > 8 {
		return nil, fmt.Errorf("bitpack: bitWidth must be 1..8, got %d", bitWidth)
	}
	totalBits := bitWidth * len(indices)
	nBytes := (totalBits + 7) / 8
	buf := make([]byte, nBytes)

	bitPos := 0
	mask := (1 << bitWidth) - 1
	for _, idx := range indices {
		if idx < 0 || idx > mask {
			return nil, fmt.Errorf("bitpack: index %d out of range for %d-bit width", idx, bitWidth)
		}
		// Write bitWidth bits starting at bitPos.
		remaining := bitWidth
		val := idx
		for remaining > 0 {
			byteIdx := bitPos / 8
			bitOff := bitPos % 8
			bitsAvail := 8 - bitOff
			bitsToWrite := remaining
			if bitsToWrite > bitsAvail {
				bitsToWrite = bitsAvail
			}
			// Extract the lowest bitsToWrite bits from val.
			chunk := val & ((1 << bitsToWrite) - 1)
			buf[byteIdx] |= byte(chunk << bitOff)
			val >>= bitsToWrite
			bitPos += bitsToWrite
			remaining -= bitsToWrite
		}
	}

	return buf, nil
}

// UnpackBits extracts n indices of bitWidth bits each from a packed byte slice.
func UnpackBits(buf []byte, bitWidth, n int) ([]int, error) {
	if bitWidth < 1 || bitWidth > 8 {
		return nil, fmt.Errorf("bitpack: bitWidth must be 1..8, got %d", bitWidth)
	}
	totalBits := bitWidth * n
	needBytes := (totalBits + 7) / 8
	if len(buf) < needBytes {
		return nil, fmt.Errorf("bitpack: buffer too small: %d bytes for %d indices at %d bits", len(buf), n, bitWidth)
	}

	indices := make([]int, n)
	mask := (1 << bitWidth) - 1
	bitPos := 0

	for i := range n {
		val := 0
		bitsRead := 0
		for bitsRead < bitWidth {
			byteIdx := bitPos / 8
			bitOff := bitPos % 8
			bitsAvail := 8 - bitOff
			bitsToRead := bitWidth - bitsRead
			if bitsToRead > bitsAvail {
				bitsToRead = bitsAvail
			}
			chunk := int(buf[byteIdx]>>bitOff) & ((1 << bitsToRead) - 1)
			val |= chunk << bitsRead
			bitPos += bitsToRead
			bitsRead += bitsToRead
		}
		indices[i] = val & mask
	}

	return indices, nil
}

// PackedSize returns the number of bytes needed to pack n indices at bitWidth bits each,
// plus 4 bytes for the stored norm.
func PackedSize(n, bitWidth int) int {
	return (bitWidth*n+7)/8 + 4
}
