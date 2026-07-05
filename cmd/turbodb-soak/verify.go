package main

import (
	"context"
	"fmt"
	"time"

	"github.com/gabrielleeyj/turbodb/internal/enginerpc"
	"github.com/gabrielleeyj/turbodb/pkg/replication"
	"github.com/jackc/pglogrepl"
)

// verifyCycle asserts the core soak invariant after a fault (or at
// baseline): with the workload quiesced, sync catches up to PostgreSQL's
// current WAL position within the budget, and a full reconcile finds zero
// discrepancies — no loss, no duplication, no divergence.
//
// On a violation the cycle repairs the divergence so subsequent cycles keep
// measuring fresh faults instead of re-reporting the same drift.
func (s *soak) verifyCycle(ctx context.Context, trigger string) bool {
	s.workload.pause()
	defer s.workload.resume()

	start := time.Now()
	ok := s.verifyQuiesced(ctx, trigger)
	if ok {
		s.board.verificationPassed()
		s.board.recordCatchup(time.Since(start))
		s.logger.Info("verification passed", "trigger", trigger, "took", time.Since(start).Round(time.Millisecond), "workload_ops", s.workload.ops.Load(), "workload_errs", s.workload.errs.Load())
	}
	return ok
}

func (s *soak) verifyQuiesced(ctx context.Context, trigger string) bool {
	// Catch-up detection uses a sentinel: with the workload quiesced,
	// capture the WAL position, then commit one marker row to the soak
	// table. Logical replication delivers transactions in commit order, so
	// the sentinel's transaction-end LSN is the first checkpointable
	// position at or beyond the capture point — once sync's checkpoint
	// reaches it, everything committed before the sentinel has been
	// applied. (Comparing against raw pg_current_wal_lsn() alone would
	// deadlock on idle streams: WAL advances for non-replicated reasons.)
	lsnText, err := s.pgQueryOne(ctx, "SELECT pg_current_wal_lsn()")
	if err != nil {
		s.board.violation("verify(%s): read wal lsn: %v", trigger, err)
		return false
	}
	target, err := pglogrepl.ParseLSN(lsnText)
	if err != nil {
		s.board.violation("verify(%s): parse lsn %q: %v", trigger, lsnText, err)
		return false
	}
	sentinel := fmt.Sprintf(
		"INSERT INTO %s VALUES ('zz-sentinel', '%s', NULL) "+
			"ON CONFLICT (doc_id) DO UPDATE SET embedding = EXCLUDED.embedding, deleted_at = NULL",
		soakTable, sentinelVector())
	if err := s.pgExec(ctx, sentinel); err != nil {
		s.board.violation("verify(%s): write sentinel: %v", trigger, err)
		return false
	}

	if err := s.waitCatchup(ctx, uint64(target)); err != nil {
		s.board.violation("verify(%s): %v", trigger, err)
		return false
	}

	report, err := s.reconcile(ctx, false)
	if err != nil {
		s.board.violation("verify(%s): reconcile: %v", trigger, err)
		return false
	}
	if report.Discrepancies() != 0 {
		s.board.violation("verify(%s): %d discrepancies (missing=%v orphaned=%v)",
			trigger, report.Discrepancies(), report.MissingInEngine, report.OrphanedInEngine)
		// Repair so later cycles start from a consistent state.
		if _, rerr := s.reconcile(ctx, true); rerr != nil {
			s.logger.Error("post-violation repair failed", "err", rerr)
		}
		return false
	}
	return true
}

// waitCatchup polls the sync checkpoint until it reaches target (the WAL
// position captured just before the sentinel write; the sentinel's
// transaction ends at or beyond it). A sync crash-restart during the window
// is fine: the supervisor restarts it and progress resumes from the
// checkpoint.
func (s *soak) waitCatchup(ctx context.Context, target uint64) error {
	deadline := time.Now().Add(s.cfg.catchupBudget)
	var last uint64
	for time.Now().Before(deadline) && ctx.Err() == nil {
		lsn, err := s.checkpointLSN()
		if err == nil {
			last = lsn
			if lsn >= target {
				return nil
			}
		}
		time.Sleep(500 * time.Millisecond)
	}
	return fmt.Errorf("sync did not catch up within %s (checkpoint=%d target=%d)",
		s.cfg.catchupBudget, last, target)
}

// reconcile runs one pass over the soak mapping.
func (s *soak) reconcile(ctx context.Context, repair bool) (replication.ReconcileReport, error) {
	cfg, err := replication.LoadConfig(s.cfg.workdir + "/sync.yaml")
	if err != nil {
		return replication.ReconcileReport{}, err
	}
	scanner, err := replication.NewPgTableScanner(ctx, s.cfg.pgDSN)
	if err != nil {
		return replication.ReconcileReport{}, err
	}
	defer func() { _ = scanner.Close() }()

	eng, err := enginerpc.Dial(engineGRPC)
	if err != nil {
		return replication.ReconcileReport{}, err
	}
	defer func() { _ = eng.Close() }()

	rec, err := replication.NewReconciler(cfg, replication.ReconcilerConfig{
		Source: scanner,
		Index:  eng,
		Engine: eng,
		Repair: repair,
	})
	if err != nil {
		return replication.ReconcileReport{}, err
	}
	return rec.ReconcileTable(ctx, cfg.Tables[0])
}

// sentinelVector renders a time-varying vector so every sentinel upsert
// emits a fresh replication event.
func sentinelVector() string {
	v := float64(time.Now().UnixNano()%1000) / 1000
	out := fmt.Sprintf("[%.3f", v)
	for i := 1; i < soakDim; i++ {
		out += ",0"
	}
	return out + "]"
}
