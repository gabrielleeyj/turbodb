// TurboDB Soak drives the Phase 6 exit criterion (SCOPE §Phase 6): a long
// fault-injection run over the hybrid deployment — PostgreSQL as source of
// truth, turbodb-sync consuming logical replication, turbodb-engine serving
// the index. It injects engine crashes, sync crashes, engine stalls
// (partition proxy), and PostgreSQL restarts, and after every fault verifies
// that reconciliation converges to zero discrepancies within the catch-up
// budget.
//
// The harness supervises both binaries the way systemd would in production:
// sync is restarted on every exit (its circuit breaker exits by design when
// the engine is unreachable), the engine after a configured outage window.
// Unexpected process deaths and failed verifications are violations; the
// run passes only with zero violations.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"
)

type soakConfig struct {
	duration      time.Duration
	faultInterval time.Duration
	workloadRate  int // ops/second
	catchupBudget time.Duration
	engineOutage  time.Duration
	stallDuration time.Duration

	pgDSN       string
	pgContainer string
	engineBin   string
	syncBin     string
	workdir     string
	seed        uint64
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "turbodb-soak: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	var cfg soakConfig
	flag.DurationVar(&cfg.duration, "duration", 24*time.Hour, "total soak duration")
	flag.DurationVar(&cfg.faultInterval, "fault-interval", 10*time.Minute, "time between injected faults")
	flag.IntVar(&cfg.workloadRate, "workload-rate", 50, "workload operations per second")
	flag.DurationVar(&cfg.catchupBudget, "catchup-budget", 2*time.Minute, "max time for sync to catch up after quiescing")
	flag.DurationVar(&cfg.engineOutage, "engine-outage", 5*time.Second, "engine downtime after a crash fault")
	flag.DurationVar(&cfg.stallDuration, "stall-duration", 15*time.Second, "engine SIGSTOP duration for the stall fault")
	flag.StringVar(&cfg.pgDSN, "pg-dsn", "postgres://postgres:turbodb@localhost:5434/postgres", "PostgreSQL DSN")
	flag.StringVar(&cfg.pgContainer, "pg-container", "turbodb-pg-soak", "docker container name for PostgreSQL restart faults")
	flag.StringVar(&cfg.engineBin, "engine-bin", "bin/turbodb-engine", "turbodb-engine binary")
	flag.StringVar(&cfg.syncBin, "sync-bin", "bin/turbodb-sync", "turbodb-sync binary")
	flag.StringVar(&cfg.workdir, "workdir", "soak-work", "working directory for data, logs, and checkpoint")
	flag.Uint64Var(&cfg.seed, "seed", 1, "fault schedule random seed")
	flag.Parse()

	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	if err := os.MkdirAll(cfg.workdir, 0o750); err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	s, err := newSoak(ctx, cfg, logger)
	if err != nil {
		return err
	}
	defer s.shutdown()

	verdict := s.runLoop(ctx, rand.New(rand.NewPCG(cfg.seed, 2))) // #nosec G404 -- reproducible fault schedule, not cryptographic
	s.board.write(filepath.Join(cfg.workdir, "scoreboard.json"))
	s.board.print(os.Stdout)
	if !verdict {
		return fmt.Errorf("SOAK FAILED: %d violation(s)", s.board.violations())
	}
	fmt.Println("SOAK PASSED")
	return nil
}

// runLoop executes fault/verify cycles until the duration elapses or ctx is
// canceled. Returns true when the run passed.
func (s *soak) runLoop(ctx context.Context, rng *rand.Rand) bool {
	deadline := time.Now().Add(s.cfg.duration)
	faults := []string{faultEngineCrash, faultSyncCrash, faultEngineStall, faultPgRestart}

	// Initial verification proves the stack is healthy before any fault.
	if !s.verifyCycle(ctx, "baseline") {
		return false
	}

	for time.Now().Before(deadline) && ctx.Err() == nil {
		// Let the workload run under normal conditions between faults.
		if !sleepCtx(ctx, s.cfg.faultInterval) {
			break
		}
		name := faults[rng.IntN(len(faults))]
		if err := s.injectFault(ctx, name); err != nil {
			s.logger.Error("fault injection failed", "fault", name, "err", err)
			s.board.violation("fault %s injection: %v", name, err)
			continue
		}
		if !s.verifyCycle(ctx, name) && ctx.Err() != nil {
			break
		}
		s.board.print(os.Stdout)
	}
	return s.board.violations() == 0 && s.board.verifications() > 0
}

// sleepCtx waits d, returning false if ctx ends first.
func sleepCtx(ctx context.Context, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-t.C:
		return true
	}
}
