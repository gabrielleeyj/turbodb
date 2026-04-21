package rotation

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestRotationPreservesNorm(t *testing.T) {
	dims := []int{64, 128, 256, 512, 1536}
	for _, d := range dims {
		rot, err := NewHadamardRotator(d, 42)
		if err != nil {
			t.Fatalf("d=%d: %v", d, err)
		}

		rng := rand.New(rand.NewPCG(123, 456)) //nolint:gosec // deterministic test
		for trial := range 1000 {
			x := randomUnitVector(rng, d)
			y := rot.Apply(x) // y has OutDim() elements

			normX := vecNorm(x)
			normY := vecNorm(y)
			if math.Abs(float64(normX-normY)) > 1e-4 {
				t.Errorf("d=%d trial=%d: ‖x‖=%f, ‖Πx‖=%f", d, trial, normX, normY)
				break
			}
		}
	}
}

func TestRotationRoundTrip(t *testing.T) {
	dims := []int{64, 128, 256, 1536}
	for _, d := range dims {
		rot, err := NewHadamardRotator(d, 99)
		if err != nil {
			t.Fatalf("d=%d: %v", d, err)
		}

		rng := rand.New(rand.NewPCG(789, 101)) //nolint:gosec // deterministic test
		for trial := range 100 {
			x := randomUnitVector(rng, d)
			y := rot.Apply(x)          // d -> padDim
			z := rot.ApplyTranspose(y) // padDim -> d

			maxErr := float32(0)
			for i := range x {
				diff := float32(math.Abs(float64(x[i] - z[i])))
				if diff > maxErr {
					maxErr = diff
				}
			}
			if maxErr > 1e-4 {
				t.Errorf("d=%d trial=%d: max roundtrip error = %e", d, trial, maxErr)
				break
			}
		}
	}
}

func TestRotationOutputDim(t *testing.T) {
	tests := []struct {
		dim, wantOut int
	}{
		{64, 64}, {128, 128}, {256, 256}, {1536, 2048}, {1000, 1024},
	}
	for _, tt := range tests {
		rot, err := NewHadamardRotator(tt.dim, 42)
		if err != nil {
			t.Fatal(err)
		}
		if rot.OutDim() != tt.wantOut {
			t.Errorf("dim=%d: OutDim()=%d, want %d", tt.dim, rot.OutDim(), tt.wantOut)
		}
		x := make([]float32, tt.dim)
		y := rot.Apply(x)
		if len(y) != tt.wantOut {
			t.Errorf("dim=%d: Apply output len=%d, want %d", tt.dim, len(y), tt.wantOut)
		}
	}
}

func TestRotationDeterminism(t *testing.T) {
	d := 1536
	seed := uint64(12345)
	x := make([]float32, d)
	rng := rand.New(rand.NewPCG(111, 222)) //nolint:gosec // deterministic test
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	rot1, _ := NewHadamardRotator(d, seed)
	y1 := rot1.Apply(x)

	rot2, _ := NewHadamardRotator(d, seed)
	y2 := rot2.Apply(x)

	for i := range y1 {
		if y1[i] != y2[i] {
			t.Fatalf("determinism failed at index %d: %f != %f", i, y1[i], y2[i])
		}
	}
}

func TestRotationDifferentSeeds(t *testing.T) {
	d := 256
	x := make([]float32, d)
	for i := range x {
		x[i] = float32(i) / float32(d)
	}

	rot1, _ := NewHadamardRotator(d, 1)
	rot2, _ := NewHadamardRotator(d, 2)

	y1 := rot1.Apply(x)
	y2 := rot2.Apply(x)

	same := true
	for i := range y1 {
		if y1[i] != y2[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("different seeds should produce different rotations")
	}
}

func TestMarshalUnmarshal(t *testing.T) {
	rot1, _ := NewHadamardRotator(1536, 42)
	data, err := rot1.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}

	rot2, err := UnmarshalRotator(data)
	if err != nil {
		t.Fatalf("UnmarshalRotator: %v", err)
	}

	if rot2.Seed() != rot1.Seed() {
		t.Errorf("seed: %d != %d", rot2.Seed(), rot1.Seed())
	}
	if rot2.Dim() != rot1.Dim() {
		t.Errorf("dim: %d != %d", rot2.Dim(), rot1.Dim())
	}

	x := make([]float32, 1536)
	for i := range x {
		x[i] = float32(i)
	}
	y1 := rot1.Apply(x)
	y2 := rot2.Apply(x)
	for i := range y1 {
		if y1[i] != y2[i] {
			t.Fatalf("output mismatch at %d after unmarshal: %f != %f", i, y1[i], y2[i])
		}
	}
}

func TestNewHadamardRotatorInvalidDim(t *testing.T) {
	_, err := NewHadamardRotator(0, 42)
	if err == nil {
		t.Error("expected error for dim=0")
	}
	_, err = NewHadamardRotator(-1, 42)
	if err == nil {
		t.Error("expected error for dim=-1")
	}
}

func TestNextPow2(t *testing.T) {
	tests := []struct {
		n, want int
	}{
		{1, 1}, {2, 2}, {3, 4}, {4, 4}, {5, 8},
		{128, 128}, {129, 256}, {1536, 2048}, {4096, 4096},
	}
	for _, tt := range tests {
		got := nextPow2(tt.n)
		if got != tt.want {
			t.Errorf("nextPow2(%d) = %d, want %d", tt.n, got, tt.want)
		}
	}
}

func BenchmarkHadamardRotator_d1536(b *testing.B) {
	rot, _ := NewHadamardRotator(1536, 42)
	rng := rand.New(rand.NewPCG(1, 2)) //nolint:gosec // deterministic bench
	x := randomUnitVector(rng, 1536)

	b.ResetTimer()
	for b.Loop() {
		_ = rot.Apply(x)
	}
}

// helpers

func randomUnitVector(rng *rand.Rand, d int) []float32 {
	x := make([]float32, d)
	var sum float64
	for i := range x {
		v := float32(rng.NormFloat64())
		x[i] = v
		sum += float64(v * v)
	}
	inv := float32(1.0 / math.Sqrt(sum))
	for i := range x {
		x[i] *= inv
	}
	return x
}

func vecNorm(x []float32) float32 {
	var sum float64
	for _, v := range x {
		sum += float64(v) * float64(v)
	}
	return float32(math.Sqrt(sum))
}
