package codebook

import (
	"sync"
	"testing"
)

func TestNewCodebookValidation(t *testing.T) {
	// Valid codebook.
	_, err := NewCodebook(128, 1, []float64{-0.5, 0.5})
	if err != nil {
		t.Errorf("valid codebook rejected: %v", err)
	}

	// Wrong number of centroids.
	_, err = NewCodebook(128, 2, []float64{-0.5, 0.5})
	if err == nil {
		t.Error("expected error for wrong centroid count")
	}

	// Unsorted centroids.
	_, err = NewCodebook(128, 1, []float64{0.5, -0.5})
	if err == nil {
		t.Error("expected error for unsorted centroids")
	}

	// Invalid dim.
	_, err = NewCodebook(0, 1, []float64{-0.5, 0.5})
	if err == nil {
		t.Error("expected error for dim=0")
	}

	// Invalid bitWidth.
	_, err = NewCodebook(128, 9, make([]float64, 512))
	if err == nil {
		t.Error("expected error for bitWidth=9")
	}
}

func TestCodebookImmutable(t *testing.T) {
	original := []float64{-0.5, 0.5}
	cb, err := NewCodebook(128, 1, original)
	if err != nil {
		t.Fatal(err)
	}

	// Mutating the input should not affect the codebook.
	original[0] = 999.0
	if cb.Centroid(0) == 999.0 {
		t.Error("codebook should not be affected by input mutation")
	}

	// Mutating Centroids() result should not affect the codebook.
	c := cb.Centroids()
	c[0] = 888.0
	if cb.Centroid(0) == 888.0 {
		t.Error("codebook should not be affected by Centroids() mutation")
	}
}

func TestNearestIndex(t *testing.T) {
	centroids := []float64{-1.5, -0.5, 0.5, 1.5}
	cb, err := NewCodebook(128, 2, centroids)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		x    float64
		want int
	}{
		{-2.0, 0},  // far left
		{-1.5, 0},  // exact match
		{-1.0, 0},  // midpoint, ties go to lower
		{-0.5, 1},  // exact match
		{0.0, 1},   // midpoint, ties go to lower
		{0.5, 2},   // exact match
		{1.0, 2},   // midpoint, ties go to lower
		{1.5, 3},   // exact match
		{100.0, 3}, // far right
	}

	for _, tt := range tests {
		got := cb.NearestIndex(tt.x)
		if got != tt.want {
			t.Errorf("NearestIndex(%f) = %d, want %d", tt.x, got, tt.want)
		}
	}
}

func TestLoadPrecomputed(t *testing.T) {
	ClearCache()
	// Load a precomputed codebook.
	cb, err := Load(1536, 4)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	if cb.Dim() != 1536 {
		t.Errorf("dim = %d, want 1536", cb.Dim())
	}
	if cb.BitWidth() != 4 {
		t.Errorf("bitWidth = %d, want 4", cb.BitWidth())
	}
	if cb.Size() != 16 {
		t.Errorf("size = %d, want 16", cb.Size())
	}

	// Verify centroids are sorted and symmetric.
	centroids := cb.Centroids()
	for i := 1; i < len(centroids); i++ {
		if centroids[i] <= centroids[i-1] {
			t.Errorf("precomputed centroids not sorted at index %d", i)
		}
	}

	// Second call should hit cache.
	cb2, err := Load(1536, 4)
	if err != nil {
		t.Fatalf("second Load failed: %v", err)
	}
	if cb2 != cb {
		t.Error("second Load should return cached codebook")
	}
}

func TestLoadOnTheFly(t *testing.T) {
	ClearCache()
	// A non-standard dim should generate on-the-fly.
	cb, err := Load(200, 2)
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	if cb.Dim() != 200 {
		t.Errorf("dim = %d, want 200", cb.Dim())
	}
	if cb.Size() != 4 {
		t.Errorf("size = %d, want 4", cb.Size())
	}
}

func TestLoadConcurrentNoRace(t *testing.T) {
	ClearCache()

	const goroutines = 50
	var wg sync.WaitGroup
	results := make([]*Codebook, goroutines)
	errs := make([]error, goroutines)

	wg.Add(goroutines)
	for i := range goroutines {
		go func(idx int) {
			defer wg.Done()
			results[idx], errs[idx] = Load(1536, 4)
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Fatalf("goroutine %d: %v", i, err)
		}
	}

	// All goroutines must get the exact same *Codebook pointer.
	for i := 1; i < goroutines; i++ {
		if results[i] != results[0] {
			t.Errorf("goroutine %d got different codebook pointer", i)
		}
	}
}

func TestLoadAllPrecomputed(t *testing.T) {
	ClearCache()
	dims := []int{128, 256, 512, 768, 1024, 1536, 3072, 4096}
	bitWidths := []int{1, 2, 3, 4, 5, 6, 8}

	for _, d := range dims {
		for _, b := range bitWidths {
			cb, err := Load(d, b)
			if err != nil {
				t.Errorf("Load(d=%d, b=%d) failed: %v", d, b, err)
				continue
			}
			if cb.Size() != 1<<b {
				t.Errorf("Load(d=%d, b=%d): size=%d, want %d", d, b, cb.Size(), 1<<b)
			}
		}
	}
}
