// Package telemetry centralizes the engine's observability surface:
//
//   - Prometheus metrics exposed at /metrics
//   - OpenTelemetry tracing (no-op unless the host installs a TracerProvider)
//   - Structured logging via log/slog (JSON in prod, text in dev)
//
// The package is engineered to be safe to use in test code without any
// configuration: a nil *Metrics is valid and turns every recording call into
// a no-op, and the OTel tracer is the global default which returns no-op
// spans until an SDK is wired in.
package telemetry

import (
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/trace"
)

const (
	// TracerName is the OpenTelemetry instrumentation scope used by the
	// engine. Hosts that wire a TracerProvider should export this scope.
	TracerName = "github.com/gabrielleeyj/turbodb"
)

// StatsSource exposes the runtime values needed by sampled gauges. The
// engine implements this interface; tests can pass any equivalent type.
type StatsSource interface {
	SegmentsActive() int
	SegmentsSealedTotal() uint64
	HostMemoryBytes() int64
	GPUMemoryBytes() int64
}

// Metrics owns the Prometheus collectors used across the engine. A nil
// receiver is valid and makes every recording call a no-op, which lets
// production code call Observe* unconditionally.
type Metrics struct {
	registry *prometheus.Registry

	searchLatency    prometheus.Histogram
	insertCount      prometheus.Counter
	segmentsSealed   prometheus.Counter
	walFsyncLatency  prometheus.Histogram
}

// Options controls metrics construction.
type Options struct {
	// Namespace prefixes every metric (defaults to "turbodb").
	Namespace string
	// Source provides the values for sampled gauges. Required.
	Source StatsSource
}

// New constructs a Metrics bundle and registers it with a fresh
// *prometheus.Registry. The registry is exposed via Handler so callers
// can mount it on an HTTP server of their choosing.
func New(opts Options) (*Metrics, error) {
	if opts.Source == nil {
		return nil, fmt.Errorf("telemetry: Options.Source must not be nil")
	}
	ns := opts.Namespace
	if ns == "" {
		ns = "turbodb"
	}

	reg := prometheus.NewRegistry()

	m := &Metrics{
		registry: reg,
		searchLatency: prometheus.NewHistogram(prometheus.HistogramOpts{
			Namespace: ns,
			Name:      "search_latency_seconds",
			Help:      "End-to-end search latency observed by the planner.",
			Buckets:   prometheus.ExponentialBuckets(0.0005, 2, 12), // 0.5ms .. ~2s
		}),
		insertCount: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: ns,
			Name:      "insert_throughput_vectors_total",
			Help:      "Total vectors inserted via Engine.Insert (durably logged).",
		}),
		segmentsSealed: prometheus.NewCounter(prometheus.CounterOpts{
			Namespace: ns,
			Name:      "segments_sealed_total",
			Help:      "Total growing segments transitioned to sealed.",
		}),
		walFsyncLatency: prometheus.NewHistogram(prometheus.HistogramOpts{
			Namespace: ns,
			Name:      "wal_fsync_latency_seconds",
			Help:      "Per-fsync latency for the write-ahead log.",
			Buckets:   prometheus.ExponentialBuckets(0.00005, 2, 14), // 50µs .. ~800ms
		}),
	}

	source := opts.Source
	collectors := []prometheus.Collector{
		m.searchLatency,
		m.insertCount,
		m.segmentsSealed,
		m.walFsyncLatency,
		prometheus.NewGaugeFunc(prometheus.GaugeOpts{
			Namespace: ns,
			Name:      "segments_active",
			Help:      "Number of segments (growing + sealed) currently held by the engine.",
		}, func() float64 { return float64(source.SegmentsActive()) }),
		prometheus.NewGaugeFunc(prometheus.GaugeOpts{
			Namespace: ns,
			Name:      "host_memory_bytes",
			Help:      "Host memory bytes pinned by sealed segments.",
		}, func() float64 { return float64(source.HostMemoryBytes()) }),
		prometheus.NewGaugeFunc(prometheus.GaugeOpts{
			Namespace: ns,
			Name:      "gpu_memory_bytes",
			Help:      "GPU memory bytes pinned by sealed segments.",
		}, func() float64 { return float64(source.GPUMemoryBytes()) }),
	}
	for _, c := range collectors {
		if err := reg.Register(c); err != nil {
			return nil, fmt.Errorf("telemetry: register collector: %w", err)
		}
	}
	return m, nil
}

// Handler returns an http.Handler that serves the Prometheus exposition
// format from the underlying registry. Mount this at /metrics.
func (m *Metrics) Handler() http.Handler {
	if m == nil {
		return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			http.Error(w, "metrics disabled", http.StatusNotImplemented)
		})
	}
	return promhttp.HandlerFor(m.registry, promhttp.HandlerOpts{})
}

// Registry exposes the underlying registry so callers can register
// additional collectors (e.g. process or runtime metrics).
func (m *Metrics) Registry() *prometheus.Registry {
	if m == nil {
		return nil
	}
	return m.registry
}

// ObserveSearchLatency records a single search end-to-end duration.
func (m *Metrics) ObserveSearchLatency(seconds float64) {
	if m == nil {
		return
	}
	m.searchLatency.Observe(seconds)
}

// AddInserts records vectors that were durably inserted.
func (m *Metrics) AddInserts(n int) {
	if m == nil || n <= 0 {
		return
	}
	m.insertCount.Add(float64(n))
}

// IncSegmentsSealed records a single growing-to-sealed transition.
func (m *Metrics) IncSegmentsSealed() {
	if m == nil {
		return
	}
	m.segmentsSealed.Inc()
}

// ObserveWALFsyncLatency records a single fsync duration in seconds.
func (m *Metrics) ObserveWALFsyncLatency(seconds float64) {
	if m == nil {
		return
	}
	m.walFsyncLatency.Observe(seconds)
}

// Tracer returns the OpenTelemetry tracer used by the engine. When no
// TracerProvider has been installed by the host, this returns a no-op
// tracer whose spans are zero-cost.
func Tracer() trace.Tracer {
	return otel.Tracer(TracerName)
}

// LogFormat selects the slog handler kind for NewLogger.
type LogFormat string

const (
	// LogFormatJSON emits machine-parseable structured logs (production).
	LogFormatJSON LogFormat = "json"
	// LogFormatText emits human-readable logs (development).
	LogFormatText LogFormat = "text"
)

// NewLogger builds a *slog.Logger writing to w with the requested format
// and level. Unknown levels fall back to Info; unknown formats fall back
// to JSON.
func NewLogger(w io.Writer, format LogFormat, level string) *slog.Logger {
	if w == nil {
		w = os.Stderr
	}
	lvl := parseLevel(level)
	opts := &slog.HandlerOptions{Level: lvl}
	switch format {
	case LogFormatText:
		return slog.New(slog.NewTextHandler(w, opts))
	default:
		return slog.New(slog.NewJSONHandler(w, opts))
	}
}

func parseLevel(s string) slog.Level {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "debug":
		return slog.LevelDebug
	case "warn", "warning":
		return slog.LevelWarn
	case "error":
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}
