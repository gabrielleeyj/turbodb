package main

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/gabrielleeyj/turbodb/pkg/index"
)

// Report captures the headline numbers from a single search pass.
type Report struct {
	Label                string
	Queries              int
	TopK                 int
	MeanRecall           float64
	MinRecall            float64
	P50Ms                float64
	P95Ms                float64
	P99Ms                float64
	MaxMs                float64
	ThroughputQPS        float64
	TotalSearchDurationS float64
}

func buildReport(label string, results [][]index.SearchResult, gt [][]int, latencies []time.Duration, topK int) Report {
	r := Report{Label: label, Queries: len(results), TopK: topK, MinRecall: 1.0}

	var sumRecall float64
	for i, res := range results {
		recall := recallAtK(res, gt[i], topK)
		sumRecall += recall
		if recall < r.MinRecall {
			r.MinRecall = recall
		}
	}
	if len(results) > 0 {
		r.MeanRecall = sumRecall / float64(len(results))
	}

	r.P50Ms = percentileMs(latencies, 0.50)
	r.P95Ms = percentileMs(latencies, 0.95)
	r.P99Ms = percentileMs(latencies, 0.99)
	r.MaxMs = percentileMs(latencies, 1.0)

	var totalNs int64
	for _, d := range latencies {
		totalNs += d.Nanoseconds()
	}
	r.TotalSearchDurationS = float64(totalNs) / 1e9
	if r.TotalSearchDurationS > 0 {
		r.ThroughputQPS = float64(len(latencies)) / r.TotalSearchDurationS
	}
	return r
}

// recallAtK returns |returned ∩ truth_topK| / k. The engine returns IDs of the
// form "vec-<index>" so we parse the numeric tail before set comparison.
func recallAtK(returned []index.SearchResult, truth []int, k int) float64 {
	if k == 0 {
		return 0
	}
	if len(truth) < k {
		k = len(truth)
	}
	want := make(map[int]struct{}, k)
	for _, idx := range truth[:k] {
		want[idx] = struct{}{}
	}

	hit := 0
	limit := len(returned)
	if limit > k {
		limit = k
	}
	for i := 0; i < limit; i++ {
		idx, ok := parseVecID(returned[i].ID)
		if !ok {
			continue
		}
		if _, present := want[idx]; present {
			hit++
		}
	}
	return float64(hit) / float64(k)
}

func parseVecID(id string) (int, bool) {
	if !strings.HasPrefix(id, "vec-") {
		return 0, false
	}
	n, err := strconv.Atoi(id[len("vec-"):])
	if err != nil {
		return 0, false
	}
	return n, true
}

func percentileMs(latencies []time.Duration, p float64) float64 {
	if len(latencies) == 0 {
		return 0
	}
	sorted := make([]time.Duration, len(latencies))
	copy(sorted, latencies)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

	idx := int(float64(len(sorted)-1)*p + 0.5)
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return float64(sorted[idx].Nanoseconds()) / 1e6
}

// PassesSLO returns true when both the recall and p99 latency targets are met.
func (r Report) PassesSLO(recallTarget, p99TargetMs float64) bool {
	return r.MeanRecall >= recallTarget && r.P99Ms <= p99TargetMs
}

// Print emits a human-readable, grep-friendly summary line per metric.
func (r Report) Print() {
	fmt.Printf("[%s] queries=%d top_k=%d\n", r.Label, r.Queries, r.TopK)
	fmt.Printf("[%s] recall mean=%.4f min=%.4f\n", r.Label, r.MeanRecall, r.MinRecall)
	fmt.Printf("[%s] latency p50=%.2fms p95=%.2fms p99=%.2fms max=%.2fms\n",
		r.Label, r.P50Ms, r.P95Ms, r.P99Ms, r.MaxMs)
	fmt.Printf("[%s] throughput=%.0f qps total=%.3fs\n",
		r.Label, r.ThroughputQPS, r.TotalSearchDurationS)
}
