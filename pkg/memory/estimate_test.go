package memory

import "testing"

func TestEstimateSegmentBytes(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name     string
		count    int
		dim      int
		bitWidth int
		want     int64
	}{
		{"zero count", 0, 128, 4, 0},
		{"zero dim", 100, 0, 4, 0},
		{"negative", -1, 128, 4, 0},
		// 128 dim * 4 bits = 64 bytes codes, +4 norm, +24 id = 92 bytes.
		// Times 100 vectors = 9200.
		{"128/4 x100", 100, 128, 4, 9200},
		// 1536 dim * 8 bits = 1536 bytes codes, +4 +24 = 1564 per vector.
		{"1536/8 x1", 1, 1536, 8, 1564},
		// Rounding: dim=10, bitWidth=3 → (30+7)/8 = 4 bytes codes.
		{"odd packing", 1, 10, 3, 4 + 4 + 24},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := EstimateSegmentBytes(tc.count, tc.dim, tc.bitWidth)
			if got != tc.want {
				t.Errorf("EstimateSegmentBytes(%d,%d,%d) = %d, want %d",
					tc.count, tc.dim, tc.bitWidth, got, tc.want)
			}
		})
	}
}
