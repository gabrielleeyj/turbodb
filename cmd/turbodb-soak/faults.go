package main

import (
	"context"
	"fmt"
	"os/exec"
	"time"

	"github.com/jackc/pgx/v5/pgconn"
)

// Fault names, also used as scoreboard keys.
const (
	faultEngineCrash = "engine-crash"
	faultSyncCrash   = "sync-crash"
	faultEngineStall = "engine-stall"
	faultPgRestart   = "pg-restart"
)

// injectFault applies one named fault and waits for the stack to recover
// (processes back up and responsive). Convergence is checked separately by
// the verification cycle.
func (s *soak) injectFault(ctx context.Context, name string) error {
	s.logger.Info("injecting fault", "fault", name)
	s.board.fault(name)

	switch name {
	case faultEngineCrash:
		s.engine.kill(s.cfg.engineOutage)
		if err := s.waitEngineReady(ctx, 60*time.Second); err != nil {
			return err
		}
	case faultSyncCrash:
		s.sync.kill(0)
		// The supervisor restarts sync; readiness shows up as checkpoint
		// progress, which the verification cycle asserts.
	case faultEngineStall:
		s.engine.stall(s.cfg.stallDuration)
		if err := s.waitEngineReady(ctx, 60*time.Second); err != nil {
			return err
		}
	case faultPgRestart:
		if out, err := exec.CommandContext(ctx, "docker", "restart", "-t", "0", s.cfg.pgContainer).CombinedOutput(); err != nil { // #nosec G204 -- container name from harness flags
			return fmt.Errorf("docker restart: %v (%s)", err, out)
		}
		if err := s.waitPostgres(ctx, 60*time.Second); err != nil {
			return err
		}
	default:
		return fmt.Errorf("unknown fault %q", name)
	}
	s.logger.Info("fault recovered", "fault", name)
	return nil
}

// waitPostgres polls until a fresh connection can run SELECT 1.
func (s *soak) waitPostgres(ctx context.Context, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) && ctx.Err() == nil {
		if err := s.pgExec(ctx, "SELECT 1"); err == nil {
			return nil
		}
		time.Sleep(500 * time.Millisecond)
	}
	return fmt.Errorf("postgres not back within %s", timeout)
}

// pgExec runs one statement on a fresh short-lived connection. The soak
// harness deliberately does not hold admin connections open: faults kill
// them constantly.
func (s *soak) pgExec(ctx context.Context, sql string) error {
	opCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	conn, err := pgconn.Connect(opCtx, s.cfg.pgDSN)
	if err != nil {
		return err
	}
	defer func() { _ = conn.Close(context.Background()) }()
	_, err = conn.Exec(opCtx, sql).ReadAll()
	return err
}

// pgQueryOne runs a single-value query on a fresh connection.
func (s *soak) pgQueryOne(ctx context.Context, sql string) (string, error) {
	opCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	conn, err := pgconn.Connect(opCtx, s.cfg.pgDSN)
	if err != nil {
		return "", err
	}
	defer func() { _ = conn.Close(context.Background()) }()
	results, err := conn.Exec(opCtx, sql).ReadAll()
	if err != nil {
		return "", err
	}
	if len(results) == 0 || len(results[0].Rows) == 0 || len(results[0].Rows[0]) == 0 {
		return "", fmt.Errorf("no rows for %q", sql)
	}
	return string(results[0].Rows[0][0]), nil
}

// setupPostgres creates the soak table and publication (idempotent; state
// survives across soak runs against the same container).
func (s *soak) setupPostgres(ctx context.Context) error {
	if err := s.waitPostgres(ctx, 30*time.Second); err != nil {
		return fmt.Errorf("postgres unreachable (start the %s container first): %w", s.cfg.pgContainer, err)
	}
	stmts := []string{
		"CREATE EXTENSION IF NOT EXISTS vector",
		fmt.Sprintf("CREATE TABLE IF NOT EXISTS %s (doc_id text PRIMARY KEY, embedding vector(%d), deleted_at text)", soakTable, soakDim),
	}
	for _, stmt := range stmts {
		if err := s.pgExec(ctx, stmt); err != nil {
			return fmt.Errorf("setup %q: %w", stmt, err)
		}
	}
	// Publication create is not IF NOT EXISTS friendly across versions;
	// tolerate duplicates.
	if err := s.pgExec(ctx, fmt.Sprintf("CREATE PUBLICATION %s FOR TABLE %s", soakPub, soakTable)); err != nil {
		if err2 := s.pgExec(ctx, fmt.Sprintf("SELECT 1 FROM pg_publication WHERE pubname = '%s'", soakPub)); err2 != nil {
			return fmt.Errorf("create publication: %w", err)
		}
	}
	return nil
}
