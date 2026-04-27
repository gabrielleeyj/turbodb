package quantizer

import (
	"context"
	"fmt"
	"runtime"
	"sync"
)

// MaxBatchSize is the maximum number of vectors that can be quantized in a
// single BatchQuantize call. This prevents unbounded memory allocation.
const MaxBatchSize = 1_000_000

// BatchQuantize quantizes multiple vectors in parallel using a worker pool
// sized to runtime.NumCPU(). The context allows callers to cancel the operation.
func BatchQuantize(ctx context.Context, q Quantizer, xs [][]float32) ([]Code, error) {
	if len(xs) == 0 {
		return nil, nil
	}
	if len(xs) > MaxBatchSize {
		return nil, fmt.Errorf("batch quantize: input size %d exceeds max batch size %d", len(xs), MaxBatchSize)
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	codes := make([]Code, len(xs))
	errs := make([]error, len(xs))

	workers := runtime.NumCPU()
	if workers > len(xs) {
		workers = len(xs)
	}

	var wg sync.WaitGroup
	work := make(chan int, len(xs))

	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range work {
				if ctx.Err() != nil {
					return
				}
				codes[i], errs[i] = q.Quantize(xs[i])
			}
		}()
	}

	for i := range xs {
		select {
		case work <- i:
		case <-ctx.Done():
			close(work)
			wg.Wait()
			return nil, ctx.Err()
		}
	}
	close(work)
	wg.Wait()

	// Return first error encountered.
	for i, err := range errs {
		if err != nil {
			return nil, fmt.Errorf("batch quantize index %d: %w", i, err)
		}
	}

	return codes, nil
}

// BatchEstimateIP computes the inner-product matrix between queries and
// database codes. Returns an N_q x N_c matrix where result[i][j] is
// the estimated inner product between queries[i] and codes[j].
func BatchEstimateIP(ctx context.Context, pq *ProdQuantizer, queries [][]float32, codes []ProdCode) ([][]float32, error) {
	if len(queries) == 0 || len(codes) == 0 {
		return nil, nil
	}

	nq := len(queries)
	nc := len(codes)
	result := make([][]float32, nq)
	for i := range result {
		result[i] = make([]float32, nc)
	}

	errs := make([]error, nq)
	workers := runtime.NumCPU()
	if workers > nq {
		workers = nq
	}

	var wg sync.WaitGroup
	work := make(chan int, nq)

	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for qi := range work {
				if ctx.Err() != nil {
					return
				}
				for ci := range nc {
					ip, err := pq.EstimateInnerProduct(queries[qi], codes[ci])
					if err != nil {
						errs[qi] = err
						return
					}
					result[qi][ci] = ip
				}
			}
		}()
	}

	for i := range nq {
		work <- i
	}
	close(work)
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			return nil, fmt.Errorf("batch estimate IP query %d: %w", i, err)
		}
	}

	return result, nil
}

// StreamQuantize reads vectors from an input channel, quantizes them, and
// sends results to an output channel. Runs until the input channel is closed
// or the context is cancelled.
//
// The caller MUST close the input channel to signal completion.
// The output channel is closed when all workers finish.
func StreamQuantize(ctx context.Context, q Quantizer, in <-chan []float32, out chan<- Code) error {
	workers := runtime.NumCPU()

	// Derive a cancellable context so the first error tears down all workers.
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	var wg sync.WaitGroup
	errCh := make(chan error, 1)

	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-ctx.Done():
					return
				case x, ok := <-in:
					if !ok {
						return
					}
					code, err := q.Quantize(x)
					if err != nil {
						select {
						case errCh <- fmt.Errorf("stream quantize: %w", err):
						default:
						}
						cancel()
						return
					}
					select {
					case out <- code:
					case <-ctx.Done():
						return
					}
				}
			}
		}()
	}

	wg.Wait()
	close(out)

	select {
	case err := <-errCh:
		return err
	default:
		if ctx.Err() != nil && ctx.Err() != context.Canceled {
			return ctx.Err()
		}
		return nil
	}
}
