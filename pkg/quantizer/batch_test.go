package quantizer

import (
	"context"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/gabrielleeyj/turbodb/pkg/codebook"
	"github.com/gabrielleeyj/turbodb/pkg/rotation"
)

func TestBatchQuantize(t *testing.T) {
	dim := 256
	bw := 4
	q := newTestMSEQuantizer(t, dim, bw)

	rng := rand.New(rand.NewPCG(42, 84)) //nolint:gosec // deterministic test
	n := 100
	xs := make([][]float32, n)
	for i := range xs {
		xs[i] = randomUnitVec(rng, dim)
	}

	codes, err := BatchQuantize(q, xs)
	if err != nil {
		t.Fatalf("BatchQuantize: %v", err)
	}

	if len(codes) != n {
		t.Fatalf("got %d codes, want %d", len(codes), n)
	}

	// Verify each code matches individual quantization.
	for i, x := range xs {
		singleCode, err := q.Quantize(x)
		if err != nil {
			t.Fatalf("Quantize[%d]: %v", i, err)
		}

		if len(codes[i].Indices) != len(singleCode.Indices) {
			t.Errorf("code[%d] size mismatch: %d vs %d", i, len(codes[i].Indices), len(singleCode.Indices))
		}
		if codes[i].Norm != singleCode.Norm {
			t.Errorf("code[%d] norm mismatch: %f vs %f", i, codes[i].Norm, singleCode.Norm)
		}
	}
}

func TestBatchQuantizeEmpty(t *testing.T) {
	dim := 256
	bw := 4
	q := newTestMSEQuantizer(t, dim, bw)

	codes, err := BatchQuantize(q, nil)
	if err != nil {
		t.Fatalf("BatchQuantize nil: %v", err)
	}
	if codes != nil {
		t.Errorf("expected nil for empty input, got %d codes", len(codes))
	}
}

func TestBatchEstimateIP(t *testing.T) {
	dim := 256
	bw := 4
	pq, _, _ := newTestProdQuantizer(t, dim, bw)

	rng := rand.New(rand.NewPCG(11, 22)) //nolint:gosec // deterministic test
	nq, nc := 5, 10

	queries := make([][]float32, nq)
	for i := range queries {
		queries[i] = randomUnitVec(rng, dim)
	}

	xs := make([][]float32, nc)
	codes := make([]ProdCode, nc)
	for i := range xs {
		xs[i] = randomUnitVec(rng, dim)
		var err error
		codes[i], err = pq.Quantize(xs[i])
		if err != nil {
			t.Fatalf("Quantize: %v", err)
		}
	}

	result, err := BatchEstimateIP(pq, queries, codes)
	if err != nil {
		t.Fatalf("BatchEstimateIP: %v", err)
	}

	if len(result) != nq {
		t.Fatalf("got %d rows, want %d", len(result), nq)
	}

	// Check that results roughly match individual estimation.
	for qi := range nq {
		if len(result[qi]) != nc {
			t.Fatalf("row %d: got %d cols, want %d", qi, len(result[qi]), nc)
		}
		for ci := range nc {
			ip, err := pq.EstimateInnerProduct(queries[qi], codes[ci])
			if err != nil {
				t.Fatal(err)
			}
			if math.Abs(float64(result[qi][ci]-ip)) > 1e-6 {
				t.Errorf("[%d][%d]: batch=%f, single=%f", qi, ci, result[qi][ci], ip)
			}
		}
	}
}

func TestStreamQuantize(t *testing.T) {
	dim := 256
	bw := 4
	q := newTestMSEQuantizer(t, dim, bw)

	rng := rand.New(rand.NewPCG(33, 44)) //nolint:gosec // deterministic test
	n := 50

	in := make(chan []float32, n)
	out := make(chan Code, n)

	// Feed input.
	go func() {
		for range n {
			in <- randomUnitVec(rng, dim)
		}
		close(in)
	}()

	err := StreamQuantize(context.Background(), q, in, out)
	if err != nil {
		t.Fatalf("StreamQuantize: %v", err)
	}

	// Count results.
	count := 0
	for range out {
		count++
	}
	if count != n {
		t.Errorf("got %d codes, want %d", count, n)
	}
}

func TestStreamQuantizeCancelStopsWorkers(t *testing.T) {
	dim := 256
	bw := 4
	q := newTestMSEQuantizer(t, dim, bw)

	rng := rand.New(rand.NewPCG(55, 66)) //nolint:gosec // deterministic test

	ctx, cancel := context.WithCancel(context.Background())
	in := make(chan []float32)
	out := make(chan Code, 100)

	done := make(chan error, 1)
	go func() {
		done <- StreamQuantize(ctx, q, in, out)
	}()

	// Send a few vectors then cancel.
	for range 5 {
		in <- randomUnitVec(rng, dim)
	}
	cancel()

	// StreamQuantize should return promptly without blocking.
	err := <-done
	// err may be nil or context.Canceled — either is acceptable.
	if err != nil {
		t.Logf("StreamQuantize returned: %v (expected)", err)
	}
}

func TestBatchQuantizeNoDataRace(t *testing.T) {
	// This test is specifically for -race detection.
	dim := 128
	bw := 2
	rot, _ := rotation.NewHadamardRotator(dim, 42)
	cb, _ := codebook.Load(dim, bw)
	q, _ := NewMSEQuantizer(dim, bw, rot, cb)

	rng := rand.New(rand.NewPCG(55, 66)) //nolint:gosec // deterministic test
	n := 1000
	xs := make([][]float32, n)
	for i := range xs {
		xs[i] = randomUnitVec(rng, dim)
	}

	codes, err := BatchQuantize(q, xs)
	if err != nil {
		t.Fatalf("BatchQuantize: %v", err)
	}
	if len(codes) != n {
		t.Fatalf("expected %d codes, got %d", n, len(codes))
	}
}

func BenchmarkBatchQuantize_1k_d256_b4(b *testing.B) {
	dim := 256
	bw := 4
	rot, _ := rotation.NewHadamardRotator(dim, 42)
	cb, _ := codebook.Load(dim, bw)
	q, _ := NewMSEQuantizer(dim, bw, rot, cb)

	rng := rand.New(rand.NewPCG(77, 88)) //nolint:gosec // deterministic bench
	n := 1000
	xs := make([][]float32, n)
	for i := range xs {
		xs[i] = randomUnitVec(rng, dim)
	}

	b.ResetTimer()
	for b.Loop() {
		_, _ = BatchQuantize(q, xs)
	}
}
