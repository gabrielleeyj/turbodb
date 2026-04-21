package quantizer

import "testing"

func TestBitPackRoundTrip(t *testing.T) {
	tests := []struct {
		name     string
		bitWidth int
		indices  []int
	}{
		{"1bit", 1, []int{0, 1, 0, 1, 1, 0, 0, 1}},
		{"2bit", 2, []int{0, 1, 2, 3, 3, 2, 1, 0}},
		{"3bit", 3, []int{0, 1, 2, 3, 4, 5, 6, 7}},
		{"4bit", 4, []int{0, 5, 10, 15, 3, 8, 12, 1}},
		{"5bit", 5, []int{0, 15, 31, 16, 7, 24}},
		{"8bit", 8, []int{0, 127, 255, 128, 1, 254}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			packed, err := PackBits(tt.indices, tt.bitWidth)
			if err != nil {
				t.Fatalf("PackBits: %v", err)
			}

			unpacked, err := UnpackBits(packed, tt.bitWidth, len(tt.indices))
			if err != nil {
				t.Fatalf("UnpackBits: %v", err)
			}

			for i := range tt.indices {
				if unpacked[i] != tt.indices[i] {
					t.Errorf("index %d: got %d, want %d", i, unpacked[i], tt.indices[i])
				}
			}
		})
	}
}

func TestBitPackSize(t *testing.T) {
	tests := []struct {
		bitWidth int
		n        int
		wantLen  int
	}{
		{4, 1536, 768},  // 4-bit, 1536 coords = 768 bytes
		{3, 1536, 576},  // 3-bit, 1536 coords = 576 bytes
		{1, 1536, 192},  // 1-bit, 1536 coords = 192 bytes
		{8, 1536, 1536}, // 8-bit = same as coord count
		{4, 2048, 1024}, // padded dimension
	}

	for _, tt := range tests {
		indices := make([]int, tt.n)
		packed, err := PackBits(indices, tt.bitWidth)
		if err != nil {
			t.Fatalf("b=%d n=%d: %v", tt.bitWidth, tt.n, err)
		}
		if len(packed) != tt.wantLen {
			t.Errorf("b=%d n=%d: packed len=%d, want %d", tt.bitWidth, tt.n, len(packed), tt.wantLen)
		}
	}
}

func TestBitPackErrors(t *testing.T) {
	_, err := PackBits([]int{0}, 0)
	if err == nil {
		t.Error("expected error for bitWidth=0")
	}

	_, err = PackBits([]int{16}, 4)
	if err == nil {
		t.Error("expected error for index out of range")
	}

	_, err = UnpackBits([]byte{0}, 4, 10)
	if err == nil {
		t.Error("expected error for buffer too small")
	}
}
