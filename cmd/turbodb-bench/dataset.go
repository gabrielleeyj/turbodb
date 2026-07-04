package main

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
	"sync"

	"github.com/gabrielleeyj/turbodb/internal/engine"
	"github.com/gabrielleeyj/turbodb/pkg/index"
)

// generateUnitVectors returns count deterministic unit-norm float32 vectors of
// the given dimension. With unit-norm inputs the inner-product metric is
// equivalent to cosine similarity, which keeps the recall comparison clean.
func generateUnitVectors(count, dim int, seed uint64) [][]float32 {
	out := make([][]float32, count)
	// Per-vector PCG seeded off (seed, i) so dataset and query streams stay
	// reproducible regardless of insert ordering or worker count.
	for i := 0; i < count; i++ {
		r := rand.New(rand.NewPCG(seed, uint64(i)+1)) // #nosec G404 -- reproducible synthetic benchmark data, not cryptographic
		v := make([]float32, dim)
		var sumSq float64
		for j := 0; j < dim; j++ {
			x := r.NormFloat64()
			v[j] = float32(x)
			sumSq += x * x
		}
		inv := float32(1.0 / math.Sqrt(sumSq))
		for j := 0; j < dim; j++ {
			v[j] *= inv
		}
		out[i] = v
	}
	return out
}

// parallelInsert dispatches inserts across workers goroutines while preserving
// deterministic IDs. Each vector is keyed as "vec-<index>" so the same dataset
// reproduces the same set of stored IDs run-to-run.
func parallelInsert(ctx context.Context, eng *engine.Engine, collection string, vectors [][]float32, workers int) error {
	jobs := make(chan int, workers*2)
	errCh := make(chan error, workers)
	var wg sync.WaitGroup

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				entry := index.VectorEntry{
					ID:     fmt.Sprintf("vec-%d", i),
					Values: vectors[i],
				}
				if err := eng.Insert(ctx, collection, entry); err != nil {
					select {
					case errCh <- fmt.Errorf("insert vec-%d: %w", i, err):
					default:
					}
					return
				}
			}
		}()
	}

	for i := range vectors {
		jobs <- i
	}
	close(jobs)
	wg.Wait()

	select {
	case err := <-errCh:
		return err
	default:
		return nil
	}
}
