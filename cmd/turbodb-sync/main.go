// TurboDB Sync is the CDC consumer that replicates data from PostgreSQL to
// the engine (SCOPE Component 7). This binary currently ships the pipeline
// plumbing: config validation and the transform/write/checkpoint loop. The
// pglogrepl source (Task 7.1) plugs into replication.EventSource and requires
// a PostgreSQL instance with logical replication enabled.
package main

import (
	"flag"
	"fmt"
	"os"

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
		return fmt.Errorf("run: the PostgreSQL logical replication source is not wired yet (Task 7.1); use check-config to validate sync.yaml")
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
