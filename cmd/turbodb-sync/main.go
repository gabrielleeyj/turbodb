// TurboDB Sync is the CDC consumer that replicates data from PostgreSQL to
// the engine (SCOPE Component 7). It consumes a logical replication slot via
// pgoutput, transforms rows per sync.yaml, and writes batched mutations to
// the engine, checkpointing the LSN after every successful flush.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/gabrielleeyj/turbodb/pkg/replication"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintf(os.Stderr, "turbodb-sync: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: turbodb-sync <check-config|run> [flags]")
	}
	switch args[0] {
	case "check-config":
		return runCheckConfig(args[1:])
	case "run":
		return runSync(args[1:])
	default:
		return fmt.Errorf("unknown command %q (expected check-config or run)", args[0])
	}
}

func runCheckConfig(args []string) error {
	fs := flag.NewFlagSet("check-config", flag.ContinueOnError)
	path := fs.String("config", "sync.yaml", "path to sync.yaml")
	if err := fs.Parse(args); err != nil {
		return err
	}
	cfg, err := replication.LoadConfig(*path)
	if err != nil {
		return err
	}
	fmt.Printf("%s: OK (%d table mapping(s))\n", *path, len(cfg.Tables))
	for _, t := range cfg.Tables {
		filter := t.Filter
		if filter == "" {
			filter = "<none>"
		}
		fmt.Printf("  %-30s -> %-15s id=%s embedding=%s filter=%s\n",
			t.Postgres, t.Engine, t.Columns.ID, t.Columns.Embedding, filter)
	}
	return nil
}

func runSync(args []string) error {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	configPath := fs.String("config", "sync.yaml", "path to sync.yaml")
	dsn := fs.String("pg-dsn", os.Getenv("TURBODB_PG_DSN"), "PostgreSQL DSN (or TURBODB_PG_DSN)")
	slot := fs.String("slot", "turbodb_sync", "replication slot name")
	publication := fs.String("publication", "turbodb_pub", "publication name")
	engineAddr := fs.String("engine", "localhost:50051", "engine gRPC address")
	checkpointPath := fs.String("checkpoint", "turbodb-sync.ckpt", "LSN checkpoint file path")
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
	cp, err := replication.NewFileCheckpoint(*checkpointPath)
	if err != nil {
		return err
	}
	startLSN, err := cp.Load()
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

	src, err := replication.NewPgSource(ctx, replication.PgSourceConfig{
		DSN:         *dsn,
		Slot:        *slot,
		Publication: *publication,
		StartLSN:    startLSN,
	})
	if err != nil {
		return err
	}
	defer func() { _ = src.Close() }()

	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	logger.Info("sync starting", "slot", *slot, "publication", *publication,
		"engine", *engineAddr, "resume_lsn", startLSN)

	w := replication.NewWriter(engine, replication.WriterConfig{})
	err = replication.Sync(ctx, src, replication.NewTransformer(cfg), w, cp,
		replication.SyncOptions{Logger: logger})
	if err != nil && ctx.Err() != nil {
		logger.Info("sync stopped by signal", "checkpointed_lsn", w.AppliedLSN())
		return nil
	}
	return err
}
