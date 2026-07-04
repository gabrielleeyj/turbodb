package main

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gabrielleeyj/turbodb/pkg/replication"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

const metricsReadTimeout = 5 * time.Second

func runReconcile(args []string) error {
	fs := flag.NewFlagSet("reconcile", flag.ContinueOnError)
	configPath := fs.String("config", "sync.yaml", "path to sync.yaml")
	dsn := fs.String("pg-dsn", os.Getenv("TURBODB_PG_DSN"), "PostgreSQL DSN (or TURBODB_PG_DSN)")
	engineAddr := fs.String("engine", "localhost:50051", "engine gRPC address")
	repair := fs.Bool("repair", false, "apply repair ops to the engine (default: report only)")
	interval := fs.Duration("interval", 0, "re-run every interval (0 = run once and exit)")
	metricsListen := fs.String("metrics-listen", "", "Prometheus HTTP listen address (empty disables; useful with --interval)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *dsn == "" {
		return fmt.Errorf("--pg-dsn (or TURBODB_PG_DSN) is required")
	}

	cfg, err := replication.LoadConfig(*configPath)
	if err != nil {
		return err
	}
	engine, err := dialEngine(*engineAddr)
	if err != nil {
		return err
	}
	defer func() { _ = engine.Close() }()

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	scanner, err := replication.NewPgTableScanner(ctx, *dsn)
	if err != nil {
		return err
	}
	defer func() { _ = scanner.Close() }()

	reg := prometheus.NewRegistry()
	metrics := replication.NewReconcileMetrics(reg)
	if *metricsListen != "" {
		mux := http.NewServeMux()
		mux.Handle("/metrics", promhttp.HandlerFor(reg, promhttp.HandlerOpts{}))
		srv := &http.Server{Addr: *metricsListen, Handler: mux, ReadHeaderTimeout: metricsReadTimeout}
		go func() { _ = srv.ListenAndServe() }()
		defer func() { _ = srv.Close() }()
	}

	rec, err := replication.NewReconciler(cfg, replication.ReconcilerConfig{
		Source:  scanner,
		Index:   engine,
		Engine:  engine,
		Repair:  *repair,
		Metrics: metrics,
	})
	if err != nil {
		return err
	}

	for {
		if err := reconcileAll(ctx, rec, cfg); err != nil {
			return err
		}
		if *interval <= 0 {
			return nil
		}
		select {
		case <-ctx.Done():
			return nil
		case <-time.After(*interval):
		}
	}
}

func reconcileAll(ctx context.Context, rec *replication.Reconciler, cfg *replication.SyncConfig) error {
	for _, mapping := range cfg.Tables {
		report, err := rec.ReconcileTable(ctx, mapping)
		if err != nil {
			return fmt.Errorf("reconcile %s: %w", mapping.Postgres, err)
		}
		printReport(mapping.Postgres, report)
	}
	return nil
}

func printReport(table string, r replication.ReconcileReport) {
	fmt.Printf("%s -> %s: source_rows=%d engine_ids=%d missing=%d orphaned=%d malformed=%d repaired=%v duration=%s\n",
		table, r.Collection, r.SourceRows, r.EngineIDs,
		len(r.MissingInEngine), len(r.OrphanedInEngine), r.MalformedRows, r.Repaired,
		r.Duration.Round(time.Millisecond))
	for _, id := range r.MissingInEngine {
		fmt.Printf("  missing in engine: %s\n", id)
	}
	for _, id := range r.OrphanedInEngine {
		fmt.Printf("  orphaned in engine: %s\n", id)
	}
}
