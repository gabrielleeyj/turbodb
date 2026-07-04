package replication

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgconn"
)

// Integration tests for PgSource against a live PostgreSQL with logical
// replication. Skipped unless TURBODB_TEST_PG_DSN is set, e.g.:
//
//	docker run -d --name turbodb-pg-test -p 5433:5432 \
//	  -e POSTGRES_PASSWORD=turbodb pgvector/pgvector:pg17 -c wal_level=logical
//	export TURBODB_TEST_PG_DSN="postgres://postgres:turbodb@localhost:5433/postgres"

func pgDSN(t *testing.T) string {
	t.Helper()
	dsn := os.Getenv("TURBODB_TEST_PG_DSN")
	if dsn == "" {
		t.Skip("TURBODB_TEST_PG_DSN not set; skipping live PostgreSQL integration test")
	}
	return dsn
}

func pgExec(t *testing.T, conn *pgconn.PgConn, sql string) {
	t.Helper()
	if _, err := conn.Exec(context.Background(), sql).ReadAll(); err != nil {
		t.Fatalf("exec %q: %v", sql, err)
	}
}

// setupPgFixture creates a fresh table + publication and returns a cleanup-
// registered admin connection. Names are unique per test to allow reruns
// against a shared database.
func setupPgFixture(t *testing.T, dsn string) (conn *pgconn.PgConn, table, pub, slot string) {
	t.Helper()
	ctx := context.Background()
	conn, err := pgconn.Connect(ctx, dsn)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	t.Cleanup(func() { _ = conn.Close(context.Background()) })

	suffix := fmt.Sprintf("%d", time.Now().UnixNano()%1_000_000_000)
	table = "sync_docs_" + suffix
	pub = "turbodb_pub_" + suffix
	slot = "turbodb_slot_" + suffix

	pgExec(t, conn, "CREATE EXTENSION IF NOT EXISTS vector")
	pgExec(t, conn, fmt.Sprintf(
		"CREATE TABLE public.%s (doc_id text PRIMARY KEY, embedding vector(3), deleted_at text)", table))
	pgExec(t, conn, fmt.Sprintf("CREATE PUBLICATION %s FOR TABLE public.%s", pub, table))
	t.Cleanup(func() {
		cctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_, _ = conn.Exec(cctx, fmt.Sprintf("SELECT pg_drop_replication_slot('%s')", slot)).ReadAll()
		_, _ = conn.Exec(cctx, fmt.Sprintf("DROP PUBLICATION IF EXISTS %s", pub)).ReadAll()
		_, _ = conn.Exec(cctx, fmt.Sprintf("DROP TABLE IF EXISTS public.%s", table)).ReadAll()
	})
	return conn, table, pub, slot
}

func syncYAMLFor(table string) []byte {
	return fmt.Appendf(nil, `
tables:
  - postgres: public.%s
    engine:   docs
    columns:
      id:        doc_id
      embedding: embedding
    filter:    "deleted_at IS NULL"
`, table)
}

// safeEngine is a threadsafe fakeEngine for use across goroutines.
type safeEngine struct {
	mu sync.Mutex
	fe fakeEngine
}

func (s *safeEngine) InsertBatch(ctx context.Context, c string, r []VectorRecord) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.fe.InsertBatch(ctx, c, r)
}

func (s *safeEngine) DeleteBatch(ctx context.Context, c string, ids []string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.fe.DeleteBatch(ctx, c, ids)
}

func (s *safeEngine) snapshot() (inserted []string, deleted []string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, batch := range s.fe.inserts {
		for _, r := range batch {
			inserted = append(inserted, r.ID)
		}
	}
	for _, batch := range s.fe.deletes {
		deleted = append(deleted, batch...)
	}
	return inserted, deleted
}

// runSyncUntil runs Sync in a goroutine until cond returns true or the
// timeout elapses, then cancels and returns Sync's error.
func runSyncUntil(t *testing.T, src EventSource, tr *Transformer, w *Writer, cp Checkpoint,
	cond func() bool, timeout time.Duration) error {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	done := make(chan error, 1)
	go func() {
		done <- Sync(ctx, src, tr, w, cp, SyncOptions{FlushInterval: 100 * time.Millisecond})
	}()

	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if !cond() {
		t.Error("condition not reached before timeout")
	}
	cancel()
	select {
	case err := <-done:
		return err
	case <-time.After(10 * time.Second):
		t.Fatal("Sync did not stop after cancel")
		return nil
	}
}

func TestPgSourceEndToEnd(t *testing.T) {
	dsn := pgDSN(t)
	admin, table, pub, slot := setupPgFixture(t, dsn)

	cfg, err := ParseConfig(syncYAMLFor(table))
	if err != nil {
		t.Fatal(err)
	}
	cp, err := NewFileCheckpoint(filepath.Join(t.TempDir(), "sync.ckpt"))
	if err != nil {
		t.Fatal(err)
	}

	src, err := NewPgSource(context.Background(), PgSourceConfig{
		DSN: dsn, Slot: slot, Publication: pub, StandbyTimeout: time.Second,
	})
	if err != nil {
		t.Fatalf("NewPgSource: %v", err)
	}
	defer func() { _ = src.Close() }()

	// Row changes after the slot exists: insert two, soft-delete one, hard-
	// delete one.
	pgExec(t, admin, fmt.Sprintf(
		"INSERT INTO public.%s VALUES ('a', '[1,2,3]', NULL), ('b', '[4,5,6]', NULL)", table))
	pgExec(t, admin, fmt.Sprintf(
		"UPDATE public.%s SET deleted_at = 'now' WHERE doc_id = 'a'", table))
	pgExec(t, admin, fmt.Sprintf("DELETE FROM public.%s WHERE doc_id = 'b'", table))

	eng := &safeEngine{}
	w := NewWriter(eng, WriterConfig{Sleep: noSleep})
	syncErr := runSyncUntil(t, src, NewTransformer(cfg), w, cp, func() bool {
		_, deleted := eng.snapshot()
		return len(deleted) >= 2
	}, 30*time.Second)
	if syncErr != nil && !errors.Is(syncErr, context.Canceled) {
		t.Fatalf("Sync: %v", syncErr)
	}

	inserted, deleted := eng.snapshot()
	if len(inserted) != 2 || inserted[0] != "a" || inserted[1] != "b" {
		t.Errorf("inserted: got %v, want [a b]", inserted)
	}
	// Soft delete of a (update failing filter) + hard delete of b.
	if len(deleted) != 2 || deleted[0] != "a" || deleted[1] != "b" {
		t.Errorf("deleted: got %v, want [a b]", deleted)
	}
	if lsn, _ := cp.Load(); lsn == 0 {
		t.Error("checkpoint LSN should have advanced")
	}
}

func TestPgSourceDurabilityAcrossRestart(t *testing.T) {
	dsn := pgDSN(t)
	admin, table, pub, slot := setupPgFixture(t, dsn)

	cfg, err := ParseConfig(syncYAMLFor(table))
	if err != nil {
		t.Fatal(err)
	}
	cp, err := NewFileCheckpoint(filepath.Join(t.TempDir(), "sync.ckpt"))
	if err != nil {
		t.Fatal(err)
	}

	// Phase 1: consume the first row, checkpoint, and stop.
	src1, err := NewPgSource(context.Background(), PgSourceConfig{
		DSN: dsn, Slot: slot, Publication: pub, StandbyTimeout: time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	pgExec(t, admin, fmt.Sprintf("INSERT INTO public.%s VALUES ('r1', '[1,1,1]', NULL)", table))

	eng1 := &safeEngine{}
	w1 := NewWriter(eng1, WriterConfig{Sleep: noSleep})
	if err := runSyncUntil(t, src1, NewTransformer(cfg), w1, cp, func() bool {
		ins, _ := eng1.snapshot()
		return len(ins) == 1
	}, 30*time.Second); err != nil && !errors.Is(err, context.Canceled) {
		t.Fatalf("phase 1 Sync: %v", err)
	}
	_ = src1.Close()

	lsn1, err := cp.Load()
	if err != nil || lsn1 == 0 {
		t.Fatalf("phase 1 checkpoint: lsn=%d err=%v", lsn1, err)
	}

	// Phase 2: new rows arrive while the consumer is down.
	pgExec(t, admin, fmt.Sprintf("INSERT INTO public.%s VALUES ('r2', '[2,2,2]', NULL)", table))

	// Restart from the checkpoint: r2 must arrive, r1 must not repeat.
	src2, err := NewPgSource(context.Background(), PgSourceConfig{
		DSN: dsn, Slot: slot, Publication: pub, StartLSN: lsn1, StandbyTimeout: time.Second,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = src2.Close() }()

	eng2 := &safeEngine{}
	w2 := NewWriter(eng2, WriterConfig{Sleep: noSleep})
	if err := runSyncUntil(t, src2, NewTransformer(cfg), w2, cp, func() bool {
		ins, _ := eng2.snapshot()
		return len(ins) >= 1
	}, 30*time.Second); err != nil && !errors.Is(err, context.Canceled) {
		t.Fatalf("phase 2 Sync: %v", err)
	}

	inserted, _ := eng2.snapshot()
	if len(inserted) != 1 || inserted[0] != "r2" {
		t.Errorf("after restart: inserted %v, want exactly [r2] (no loss, no duplication)", inserted)
	}
	if lsn2, _ := cp.Load(); lsn2 <= lsn1 {
		t.Errorf("checkpoint should advance: %d -> %d", lsn1, lsn2)
	}
}
