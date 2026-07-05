package main

import (
	"context"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/jackc/pgx/v5/pgconn"
)

// workloadIDSpace bounds the number of distinct ids so the table stays a
// constant size over arbitrarily long runs (ops recycle ids).
const workloadIDSpace = 20_000

// workload issues a steady mix of upserts, soft deletes, undeletes, and
// hard deletes against the soak table, reconnecting through PostgreSQL
// restarts. It can be paused for verification cycles.
type workload struct {
	dsn    string
	rate   int
	logger *slog.Logger

	paused     atomic.Bool
	ops        atomic.Int64
	reconnects atomic.Int64
	errs       atomic.Int64

	mu     sync.Mutex
	conn   *pgconn.PgConn
	cancel context.CancelFunc
	done   chan struct{}
}

func newWorkload(dsn string, rate int, logger *slog.Logger) *workload {
	return &workload{dsn: dsn, rate: rate, logger: logger, done: make(chan struct{})}
}

func (w *workload) start(parent context.Context) {
	ctx, cancel := context.WithCancel(parent)
	w.cancel = cancel
	go w.loop(ctx)
}

// pause blocks new operations and waits for the in-flight one to finish.
// Holding mu briefly is a barrier: execOne runs under the same mutex, so
// once acquired, no operation is mid-flight.
func (w *workload) pause() {
	w.paused.Store(true)
	w.mu.Lock()
	defer w.mu.Unlock() //nolint:staticcheck // SA2001: mutex as in-flight-op barrier
}

func (w *workload) resume() { w.paused.Store(false) }

func (w *workload) loop(ctx context.Context) {
	defer close(w.done)
	rng := rand.New(rand.NewPCG(7, 7)) // #nosec G404 -- synthetic workload data
	interval := time.Second / time.Duration(w.rate)
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			w.closeConn()
			return
		case <-ticker.C:
		}
		if w.paused.Load() {
			continue
		}
		w.mu.Lock()
		w.execOne(ctx, rng)
		w.mu.Unlock()
	}
}

// execOne performs one operation, reconnecting on failure (PostgreSQL may
// be mid-restart; those errors are expected turbulence, not violations).
func (w *workload) execOne(ctx context.Context, rng *rand.Rand) {
	if w.conn == nil {
		conn, err := pgconn.Connect(ctx, w.dsn)
		if err != nil {
			// Expected during PostgreSQL restart faults; retry next tick.
			if w.errs.Add(1)%100 == 1 {
				w.logger.Warn("workload connect failed", "err", err)
			}
			return
		}
		w.conn = conn
		w.reconnects.Add(1)
	}

	id := fmt.Sprintf("soak-%06d", rng.IntN(workloadIDSpace))
	var sql string
	var args [][]byte
	switch r := rng.IntN(100); {
	case r < 70:
		sql = "INSERT INTO " + soakTable + " VALUES ($1, $2, NULL) " +
			"ON CONFLICT (doc_id) DO UPDATE SET embedding = EXCLUDED.embedding, deleted_at = NULL"
		args = [][]byte{[]byte(id), []byte(randomVectorText(rng))}
	case r < 85:
		sql = "UPDATE " + soakTable + " SET deleted_at = 'soaked' WHERE doc_id = $1"
		args = [][]byte{[]byte(id)}
	case r < 95:
		sql = "UPDATE " + soakTable + " SET deleted_at = NULL WHERE doc_id = $1"
		args = [][]byte{[]byte(id)}
	default:
		sql = "DELETE FROM " + soakTable + " WHERE doc_id = $1"
		args = [][]byte{[]byte(id)}
	}

	opCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	_, err := w.conn.ExecParams(opCtx, sql, args, nil, nil, nil).Close()
	cancel()
	if err != nil {
		if ctx.Err() == nil {
			if w.errs.Add(1)%100 == 1 {
				w.logger.Warn("workload op failed", "err", err)
			}
			w.closeConn() // force reconnect next tick
		}
		return
	}
	w.ops.Add(1)
}

func (w *workload) closeConn() {
	if w.conn != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		_ = w.conn.Close(ctx)
		cancel()
		w.conn = nil
	}
}

// stop cancels the workload and waits for the loop to exit.
func (w *workload) stop() {
	if w.cancel != nil {
		w.cancel()
		<-w.done
	}
}

// randomVectorText renders a pgvector literal like "[0.1,0.2,...]".
func randomVectorText(rng *rand.Rand) string {
	var b strings.Builder
	b.WriteByte('[')
	for i := 0; i < soakDim; i++ {
		if i > 0 {
			b.WriteByte(',')
		}
		fmt.Fprintf(&b, "%.4f", rng.Float64()*2-1)
	}
	b.WriteByte(']')
	return b.String()
}
