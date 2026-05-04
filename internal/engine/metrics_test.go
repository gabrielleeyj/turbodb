package engine

import (
	"io"
	"math/rand/v2"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gabrielleeyj/turbodb/pkg/index"
	"github.com/gabrielleeyj/turbodb/pkg/search"
	"github.com/gabrielleeyj/turbodb/pkg/telemetry"
)

// TestMetricsExposition exercises the full observability path: build the
// engine, attach Prometheus metrics, drive insert/search traffic, then
// scrape /metrics and assert the counters/histograms reflect activity.
func TestMetricsExposition(t *testing.T) {
	t.Parallel()
	e := newTestEngine(t)

	metrics, err := telemetry.New(telemetry.Options{Source: e})
	if err != nil {
		t.Fatalf("telemetry.New: %v", err)
	}
	e.AttachMetrics(metrics)

	if err := e.CreateCollection(t.Context(), defaultCollection("metrics")); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	rng := rand.New(rand.NewPCG(1, 2))
	const inserts = 16
	for i := 0; i < inserts; i++ {
		if err := e.Insert(t.Context(), "metrics", index.VectorEntry{
			ID:     "v" + string(rune('a'+i)),
			Values: randVec(rng, testDim),
		}); err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}

	if _, _, err := e.Search(t.Context(), "metrics", randVec(rng, testDim), search.Options{TopK: 4}); err != nil {
		t.Fatalf("Search: %v", err)
	}

	srv := httptest.NewServer(metrics.Handler())
	defer srv.Close()

	resp, err := http.Get(srv.URL)
	if err != nil {
		t.Fatalf("scrape: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("scrape status: got %d, want 200", resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	exposition := string(body)

	wantSubstrings := []string{
		"turbodb_insert_throughput_vectors_total 16",
		"turbodb_search_latency_seconds_count 1",
		"turbodb_segments_active",
		"turbodb_host_memory_bytes",
		"turbodb_gpu_memory_bytes",
		"turbodb_wal_fsync_latency_seconds_count",
	}
	for _, want := range wantSubstrings {
		if !strings.Contains(exposition, want) {
			t.Errorf("metrics output missing %q\n--- output ---\n%s", want, exposition)
		}
	}
}

// TestMetricsHandlerNilSafe verifies the engine path stays nil-safe when no
// Metrics are attached (the production default before main.go wires them).
func TestMetricsHandlerNilSafe(t *testing.T) {
	t.Parallel()
	e := newTestEngine(t)
	if err := e.CreateCollection(t.Context(), defaultCollection("nilmetrics")); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	if err := e.Insert(t.Context(), "nilmetrics", index.VectorEntry{
		ID:     "v0",
		Values: randVec(rand.New(rand.NewPCG(7, 8)), testDim),
	}); err != nil {
		t.Fatalf("Insert: %v", err)
	}
}
