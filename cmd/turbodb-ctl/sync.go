package main

import (
	"fmt"
	"os"
	"time"

	"github.com/gabrielleeyj/turbodb/internal/enginerpc"
	"github.com/gabrielleeyj/turbodb/pkg/replication"
	"github.com/spf13/cobra"
)

func newSyncCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "sync",
		Short: "Inspect and reconcile CDC replication state",
	}
	cmd.AddCommand(newSyncStatusCmd(), newSyncReconcileCmd())
	return cmd
}

func newSyncStatusCmd() *cobra.Command {
	var checkpointPath string
	cmd := &cobra.Command{
		Use:   "status",
		Short: "Show the sync checkpoint position",
		RunE: func(cmd *cobra.Command, _ []string) error {
			cp, err := replication.NewFileCheckpoint(checkpointPath)
			if err != nil {
				return err
			}
			lsn, err := cp.Load()
			if err != nil {
				return err
			}
			out := cmd.OutOrStdout()
			if lsn == 0 {
				fmt.Fprintf(out, "%s: no checkpoint (sync has not committed any position yet)\n", checkpointPath)
				return nil
			}
			// Render in PostgreSQL's XXX/XXX LSN notation alongside the raw value.
			fmt.Fprintf(out, "%s: lsn=%d (%X/%X)\n", checkpointPath, lsn, uint32(lsn>>32), uint32(lsn)) // #nosec G115 -- intentional LSN halves
			return nil
		},
	}
	cmd.Flags().StringVar(&checkpointPath, "checkpoint", "turbodb-sync.ckpt", "LSN checkpoint file path")
	return cmd
}

func newSyncReconcileCmd() *cobra.Command {
	var (
		configPath string
		dsn        string
		repair     bool
	)
	cmd := &cobra.Command{
		Use:   "reconcile [collection]",
		Short: "Diff PostgreSQL against the engine and optionally repair",
		Long: "Reconcile merge-diffs each configured source table against its engine " +
			"collection by id. With a collection argument, only that mapping runs. " +
			"Repairs are only applied with --repair.",
		Args: cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			if dsn == "" {
				return fmt.Errorf("--pg-dsn (or TURBODB_PG_DSN) is required")
			}
			cfg, err := replication.LoadConfig(configPath)
			if err != nil {
				return err
			}
			mappings := cfg.Tables
			if len(args) == 1 {
				mappings = nil
				for _, m := range cfg.Tables {
					if m.Engine == args[0] {
						mappings = append(mappings, m)
					}
				}
				if len(mappings) == 0 {
					return fmt.Errorf("no table mapping for collection %q in %s", args[0], configPath)
				}
			}

			engineAddr, err := cmd.Flags().GetString("engine")
			if err != nil {
				return err
			}
			eng, err := enginerpc.Dial(engineAddr)
			if err != nil {
				return err
			}
			defer func() { _ = eng.Close() }()

			ctx := cmd.Context()
			scanner, err := replication.NewPgTableScanner(ctx, dsn)
			if err != nil {
				return err
			}
			defer func() { _ = scanner.Close() }()

			rec, err := replication.NewReconciler(cfg, replication.ReconcilerConfig{
				Source: scanner,
				Index:  eng,
				Engine: eng,
				Repair: repair,
			})
			if err != nil {
				return err
			}
			for _, mapping := range mappings {
				report, err := rec.ReconcileTable(ctx, mapping)
				if err != nil {
					return fmt.Errorf("reconcile %s: %w", mapping.Postgres, err)
				}
				printReconcileReport(cmd, mapping.Postgres, report)
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&configPath, "config", "sync.yaml", "path to sync.yaml")
	cmd.Flags().StringVar(&dsn, "pg-dsn", os.Getenv("TURBODB_PG_DSN"), "PostgreSQL DSN (or TURBODB_PG_DSN)")
	cmd.Flags().BoolVar(&repair, "repair", false, "apply repair ops to the engine")
	return cmd
}

func printReconcileReport(cmd *cobra.Command, table string, r replication.ReconcileReport) {
	out := cmd.OutOrStdout()
	fmt.Fprintf(out, "%s -> %s: source_rows=%d engine_ids=%d missing=%d orphaned=%d malformed=%d repaired=%v duration=%s\n",
		table, r.Collection, r.SourceRows, r.EngineIDs,
		len(r.MissingInEngine), len(r.OrphanedInEngine), r.MalformedRows, r.Repaired,
		r.Duration.Round(time.Millisecond))
	for _, id := range r.MissingInEngine {
		fmt.Fprintf(out, "  missing in engine: %s\n", id)
	}
	for _, id := range r.OrphanedInEngine {
		fmt.Fprintf(out, "  orphaned in engine: %s\n", id)
	}
}
