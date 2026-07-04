// Command turbodb-bench validates Phase 3 exit criteria for the standalone
// engine: load N vectors, measure recall@k against brute-force ground truth,
// report p50/p95/p99 search latency, and (optionally) verify that the same
// recall holds after a close/reopen cycle (crash-recovery).
//
// The exit criteria from SCOPE.md §17 (Phase 3) are:
//
//	100k-vector collection served via the engine,
//	recall ≥ 0.95, p99 < 20ms, survives crash-recover.
//
// Defaults are sized for a laptop run; pass --vectors 100000 to exercise the
// full SCOPE target.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"time"

	"github.com/gabrielleeyj/turbodb/internal/engine"
	"github.com/gabrielleeyj/turbodb/pkg/index"
	"github.com/gabrielleeyj/turbodb/pkg/search"
)

type config struct {
	dataDir    string
	vectors    int
	queries    int
	dim        int
	bitWidth   int
	topK       int
	oversearch float64
	rerank     bool
	seed       uint64
	flush      bool
	crashTest  bool
	keepData   bool

	recallTarget  float64
	p99TargetMs   float64
	insertWorkers int
	sealThreshold int
}

func main() {
	cfg := parseFlags()
	if err := run(cfg); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func parseFlags() config {
	cfg := config{}
	flag.StringVar(&cfg.dataDir, "data-dir", "", "engine data directory (default: temp dir, removed on exit)")
	flag.IntVar(&cfg.vectors, "vectors", 10000, "number of vectors to load (use 100000 for the SCOPE phase-3 target)")
	flag.IntVar(&cfg.queries, "queries", 200, "number of search queries to run")
	flag.IntVar(&cfg.dim, "dim", 256, "vector dimension")
	flag.IntVar(&cfg.bitWidth, "bit-width", 4, "MSE quantizer bit width (1..8)")
	flag.IntVar(&cfg.topK, "top-k", 10, "top-k for recall and search")
	flag.Float64Var(&cfg.oversearch, "oversearch", 1.0, "planner oversearch factor (>=1.0)")
	flag.BoolVar(&cfg.rerank, "rerank", false, "enable planner rerank flag")
	flag.Uint64Var(&cfg.seed, "seed", 42, "deterministic RNG seed")
	flag.BoolVar(&cfg.flush, "flush", true, "flush before search to seal the active segment")
	flag.BoolVar(&cfg.crashTest, "crash-recover", false, "close and reopen the engine before search to validate WAL recovery")
	flag.BoolVar(&cfg.keepData, "keep-data", false, "do not remove the data directory on exit")
	flag.Float64Var(&cfg.recallTarget, "recall-target", 0.95, "fail if mean recall@k drops below this")
	flag.Float64Var(&cfg.p99TargetMs, "p99-target-ms", 20.0, "fail if p99 search latency exceeds this many milliseconds")
	flag.IntVar(&cfg.insertWorkers, "insert-workers", 4, "concurrent insert workers")
	flag.IntVar(&cfg.sealThreshold, "seal-threshold", 0, "vectors per growing segment before auto-seal (0 = engine default)")
	flag.Parse()
	return cfg
}

func run(cfg config) error {
	if cfg.vectors < 1 {
		return fmt.Errorf("vectors must be >= 1")
	}
	if cfg.queries < 1 {
		return fmt.Errorf("queries must be >= 1")
	}
	if cfg.topK < 1 {
		return fmt.Errorf("top-k must be >= 1")
	}
	if cfg.topK > cfg.vectors {
		return fmt.Errorf("top-k (%d) must be <= vectors (%d)", cfg.topK, cfg.vectors)
	}
	if cfg.insertWorkers < 1 {
		cfg.insertWorkers = 1
	}

	dataDir, cleanup, err := resolveDataDir(cfg)
	if err != nil {
		return err
	}
	defer cleanup()

	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelWarn}))

	fmt.Printf("turbodb-bench: vectors=%d dim=%d bit_width=%d top_k=%d queries=%d data=%s\n",
		cfg.vectors, cfg.dim, cfg.bitWidth, cfg.topK, cfg.queries, dataDir)

	collection := "bench"

	rawVectors := generateUnitVectors(cfg.vectors, cfg.dim, cfg.seed)
	queries := generateUnitVectors(cfg.queries, cfg.dim, cfg.seed^0x9E3779B97F4A7C15)

	gt, gtElapsed := computeGroundTruth(rawVectors, queries, cfg.topK)
	fmt.Printf("ground truth: %d queries × %d candidates in %s\n",
		cfg.queries, cfg.vectors, gtElapsed.Round(time.Millisecond))

	loadElapsed, err := loadEngine(dataDir, logger, collection, cfg, rawVectors)
	if err != nil {
		return fmt.Errorf("load: %w", err)
	}
	fmt.Printf("load: inserted %d vectors in %s (%.0f vec/s)\n",
		cfg.vectors, loadElapsed.Round(time.Millisecond),
		float64(cfg.vectors)/loadElapsed.Seconds())

	report, err := runSearchPass(dataDir, logger, collection, cfg, queries, gt, "search")
	if err != nil {
		return err
	}
	if cfg.crashTest {
		fmt.Println("crash-recover: closing and reopening engine before second search pass")
		report2, err := runSearchPass(dataDir, logger, collection, cfg, queries, gt, "search-after-recover")
		if err != nil {
			return err
		}
		// The post-recovery report is the one we judge against the SLO so that a
		// regression introduced by replay is caught.
		report = report2
	}

	if !report.PassesSLO(cfg.recallTarget, cfg.p99TargetMs) {
		return fmt.Errorf("phase-3 exit criteria NOT met (recall>=%.3f and p99<=%.1fms)", cfg.recallTarget, cfg.p99TargetMs)
	}
	fmt.Printf("phase-3 exit criteria met: recall@%d=%.4f p99=%.2fms\n", cfg.topK, report.MeanRecall, report.P99Ms)
	return nil
}

func resolveDataDir(cfg config) (string, func(), error) {
	if cfg.dataDir != "" {
		if err := os.MkdirAll(cfg.dataDir, 0o750); err != nil {
			return "", nil, fmt.Errorf("create data dir: %w", err)
		}
		return cfg.dataDir, func() {}, nil
	}
	dir, err := os.MkdirTemp("", "turbodb-bench-*")
	if err != nil {
		return "", nil, fmt.Errorf("create temp dir: %w", err)
	}
	cleanup := func() {
		if cfg.keepData {
			fmt.Fprintf(os.Stderr, "data retained at %s\n", dir)
			return
		}
		_ = os.RemoveAll(dir)
	}
	return dir, cleanup, nil
}

func loadEngine(dataDir string, logger *slog.Logger, collection string, cfg config, vectors [][]float32) (time.Duration, error) {
	eng, err := engine.New(engine.Config{
		DataDir:       dataDir,
		Logger:        logger,
		SealThreshold: cfg.sealThreshold,
	})
	if err != nil {
		return 0, err
	}
	defer func() { _ = eng.Close() }()

	ctx := context.Background()
	if err := eng.CreateCollection(ctx, engine.CollectionConfig{
		Name:        collection,
		Dim:         cfg.dim,
		BitWidth:    cfg.bitWidth,
		Metric:      engine.MetricInnerProduct,
		Variant:     engine.VariantMSE,
		RotatorSeed: cfg.seed,
	}); err != nil {
		return 0, err
	}

	start := time.Now()
	if err := parallelInsert(ctx, eng, collection, vectors, cfg.insertWorkers); err != nil {
		return 0, err
	}
	if cfg.flush {
		if err := eng.Flush(ctx, collection); err != nil {
			return 0, fmt.Errorf("flush: %w", err)
		}
	}
	elapsed := time.Since(start)

	stats, err := eng.Stats(collection)
	if err == nil {
		fmt.Printf("collection stats: total=%d sealed_segments=%d growing=%d pinned_bytes=%d\n",
			stats.VectorCount, stats.SealedSegmentCount, stats.GrowingSegmentCount, stats.PinnedBytes)
	}
	return elapsed, nil
}

func runSearchPass(dataDir string, logger *slog.Logger, collection string, cfg config, queries [][]float32, gt [][]int, label string) (Report, error) {
	eng, err := engine.New(engine.Config{
		DataDir:       dataDir,
		Logger:        logger,
		SealThreshold: cfg.sealThreshold,
	})
	if err != nil {
		return Report{}, err
	}
	defer func() { _ = eng.Close() }()

	ctx := context.Background()
	opts := search.Options{
		TopK:             cfg.topK,
		OversearchFactor: cfg.oversearch,
		Rerank:           cfg.rerank,
	}

	latencies := make([]time.Duration, len(queries))
	results := make([][]index.SearchResult, len(queries))

	for i, q := range queries {
		t0 := time.Now()
		res, _, err := eng.Search(ctx, collection, q, opts)
		latencies[i] = time.Since(t0)
		if err != nil {
			return Report{}, fmt.Errorf("query %d: %w", i, err)
		}
		results[i] = res
	}

	report := buildReport(label, results, gt, latencies, cfg.topK)
	report.Print()
	return report, nil
}
