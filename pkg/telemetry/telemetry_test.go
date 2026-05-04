package telemetry

import (
	"bytes"
	"context"
	"io"
	"log/slog"
	"net/http/httptest"
	"strings"
	"testing"
)

type fakeStats struct {
	active   int
	sealed   uint64
	hostMem  int64
	gpuMem   int64
}

func (f *fakeStats) SegmentsActive() int        { return f.active }
func (f *fakeStats) SegmentsSealedTotal() uint64 { return f.sealed }
func (f *fakeStats) HostMemoryBytes() int64     { return f.hostMem }
func (f *fakeStats) GPUMemoryBytes() int64      { return f.gpuMem }

func TestNewRequiresSource(t *testing.T) {
	t.Parallel()
	if _, err := New(Options{}); err == nil {
		t.Fatalf("expected error when Source is nil")
	}
}

func TestNilMetricsIsNoop(t *testing.T) {
	t.Parallel()
	var m *Metrics
	// None of these should panic.
	m.ObserveSearchLatency(0.1)
	m.AddInserts(10)
	m.IncSegmentsSealed()
	m.ObserveWALFsyncLatency(0.001)

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/metrics", nil)
	m.Handler().ServeHTTP(rec, req)
	if rec.Code != 501 {
		t.Errorf("nil handler status = %d, want 501", rec.Code)
	}
	if m.Registry() != nil {
		t.Errorf("nil Registry should be nil")
	}
}

func TestMetricsExposition(t *testing.T) {
	t.Parallel()
	src := &fakeStats{active: 3, sealed: 7, hostMem: 1024, gpuMem: 2048}
	m, err := New(Options{Source: src})
	if err != nil {
		t.Fatal(err)
	}

	m.ObserveSearchLatency(0.001)
	m.ObserveSearchLatency(0.005)
	m.AddInserts(42)
	m.IncSegmentsSealed()
	m.IncSegmentsSealed()
	m.ObserveWALFsyncLatency(0.0001)

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/metrics", nil)
	m.Handler().ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("Handler status = %d, want 200", rec.Code)
	}
	body := rec.Body.String()

	mustContain(t, body, "turbodb_search_latency_seconds_count 2")
	mustContain(t, body, "turbodb_insert_throughput_vectors_total 42")
	mustContain(t, body, "turbodb_segments_sealed_total 2")
	mustContain(t, body, "turbodb_wal_fsync_latency_seconds_count 1")
	mustContain(t, body, "turbodb_segments_active 3")
	mustContain(t, body, "turbodb_host_memory_bytes 1024")
	mustContain(t, body, "turbodb_gpu_memory_bytes 2048")
}

func TestGaugeFuncReflectsLiveState(t *testing.T) {
	t.Parallel()
	src := &fakeStats{active: 1}
	m, err := New(Options{Source: src})
	if err != nil {
		t.Fatal(err)
	}

	src.active = 9
	src.hostMem = 5000

	rec := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/metrics", nil)
	m.Handler().ServeHTTP(rec, req)
	body := rec.Body.String()
	mustContain(t, body, "turbodb_segments_active 9")
	mustContain(t, body, "turbodb_host_memory_bytes 5000")
}

func TestNewLoggerJSON(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	logger := NewLogger(&buf, LogFormatJSON, "debug")
	logger.Info("hello", slog.String("k", "v"))
	out := buf.String()
	if !strings.Contains(out, `"msg":"hello"`) {
		t.Errorf("expected JSON output, got %q", out)
	}
	if !strings.Contains(out, `"k":"v"`) {
		t.Errorf("missing attribute, got %q", out)
	}
}

func TestNewLoggerText(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	logger := NewLogger(&buf, LogFormatText, "info")
	logger.Info("hello", slog.String("k", "v"))
	out := buf.String()
	if !strings.Contains(out, "msg=hello") || !strings.Contains(out, "k=v") {
		t.Errorf("unexpected text output: %q", out)
	}
}

func TestNewLoggerLevelFilter(t *testing.T) {
	t.Parallel()
	var buf bytes.Buffer
	logger := NewLogger(&buf, LogFormatJSON, "warn")
	logger.Info("filtered")
	logger.Warn("kept")
	out := buf.String()
	if strings.Contains(out, "filtered") {
		t.Errorf("info log leaked through warn filter: %q", out)
	}
	if !strings.Contains(out, "kept") {
		t.Errorf("warn log dropped: %q", out)
	}
}

func TestNewLoggerNilWriterFallsBack(t *testing.T) {
	t.Parallel()
	// nil writer should default to os.Stderr; just exercise the path
	// without panicking.
	logger := NewLogger(nil, LogFormatJSON, "")
	logger.Info("ok")
}

func TestTracerReturnsNoopByDefault(t *testing.T) {
	t.Parallel()
	tr := Tracer()
	if tr == nil {
		t.Fatal("Tracer() returned nil")
	}
	_, span := tr.Start(context.Background(), "noop")
	defer span.End()
	if span.SpanContext().HasTraceID() {
		t.Errorf("expected no-op span (no trace id), got real one")
	}
}

func mustContain(t *testing.T, haystack, needle string) {
	t.Helper()
	if !strings.Contains(haystack, needle) {
		t.Errorf("metrics output missing %q\nfull output:\n%s", needle, haystack)
	}
}

// Ensure io.Writer is referenced (avoids unused import in some build configs).
var _ io.Writer = (*bytes.Buffer)(nil)
