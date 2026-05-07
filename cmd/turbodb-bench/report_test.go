package main

import (
	"math"
	"testing"
	"time"

	"github.com/gabrielleeyj/turbodb/pkg/index"
)

func TestRecallAtK(t *testing.T) {
	tests := []struct {
		name     string
		returned []string
		truth    []int
		k        int
		want     float64
	}{
		{"perfect", []string{"vec-1", "vec-2", "vec-3"}, []int{1, 2, 3}, 3, 1.0},
		{"none", []string{"vec-7", "vec-8", "vec-9"}, []int{1, 2, 3}, 3, 0.0},
		{"partial", []string{"vec-1", "vec-9", "vec-3"}, []int{1, 2, 3}, 3, 2.0 / 3.0},
		{"truncated", []string{"vec-1"}, []int{1, 2, 3}, 3, 1.0 / 3.0},
		{"unknown id", []string{"oops", "vec-1"}, []int{1, 2}, 2, 0.5},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			res := make([]index.SearchResult, len(tc.returned))
			for i, id := range tc.returned {
				res[i] = index.SearchResult{ID: id}
			}
			got := recallAtK(res, tc.truth, tc.k)
			if math.Abs(got-tc.want) > 1e-9 {
				t.Fatalf("recallAtK = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestPercentileMs(t *testing.T) {
	lats := []time.Duration{
		1 * time.Millisecond,
		2 * time.Millisecond,
		3 * time.Millisecond,
		4 * time.Millisecond,
		5 * time.Millisecond,
		6 * time.Millisecond,
		7 * time.Millisecond,
		8 * time.Millisecond,
		9 * time.Millisecond,
		10 * time.Millisecond,
	}
	if p99 := percentileMs(lats, 0.99); p99 != 10 {
		t.Fatalf("p99 = %v, want 10", p99)
	}
	if p50 := percentileMs(lats, 0.50); math.Abs(p50-6) > 1e-9 && math.Abs(p50-5) > 1e-9 {
		t.Fatalf("p50 = %v, want ~5 or 6", p50)
	}
	if p := percentileMs(nil, 0.5); p != 0 {
		t.Fatalf("empty p = %v, want 0", p)
	}
}

func TestUnitVectorsAreNormalized(t *testing.T) {
	vs := generateUnitVectors(8, 32, 1234)
	for i, v := range vs {
		var sumSq float64
		for _, x := range v {
			sumSq += float64(x) * float64(x)
		}
		if math.Abs(math.Sqrt(sumSq)-1.0) > 1e-4 {
			t.Fatalf("vec %d: norm = %v, want ~1.0", i, math.Sqrt(sumSq))
		}
	}
}

func TestGenerateUnitVectorsDeterministic(t *testing.T) {
	a := generateUnitVectors(4, 16, 99)
	b := generateUnitVectors(4, 16, 99)
	for i := range a {
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				t.Fatalf("nondeterministic at [%d][%d]: %v vs %v", i, j, a[i][j], b[i][j])
			}
		}
	}
}
