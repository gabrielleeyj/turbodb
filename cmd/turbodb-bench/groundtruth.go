package main

import (
	"runtime"
	"sort"
	"sync"
	"time"
)

// computeGroundTruth returns, for each query, the indices of the topK most
// similar dataset vectors by inner product. With unit-norm inputs this is
// equivalent to cosine similarity. The brute-force pass is parallelized across
// GOMAXPROCS workers to keep wall-time reasonable on laptops.
func computeGroundTruth(dataset, queries [][]float32, topK int) ([][]int, time.Duration) {
	out := make([][]int, len(queries))
	start := time.Now()

	workers := runtime.GOMAXPROCS(0)
	if workers > len(queries) {
		workers = len(queries)
	}
	jobs := make(chan int, workers*2)
	var wg sync.WaitGroup

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for q := range jobs {
				out[q] = bruteForceTopK(dataset, queries[q], topK)
			}
		}()
	}
	for q := range queries {
		jobs <- q
	}
	close(jobs)
	wg.Wait()

	return out, time.Since(start)
}

func bruteForceTopK(dataset [][]float32, query []float32, topK int) []int {
	type scored struct {
		idx   int
		score float32
	}
	all := make([]scored, len(dataset))
	for i, v := range dataset {
		all[i] = scored{idx: i, score: dot(query, v)}
	}
	// Partial sort would suffice but the dataset is bounded by the bench size.
	sort.Slice(all, func(i, j int) bool { return all[i].score > all[j].score })
	if topK > len(all) {
		topK = len(all)
	}
	ids := make([]int, topK)
	for i := 0; i < topK; i++ {
		ids[i] = all[i].idx
	}
	return ids
}

func dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}
