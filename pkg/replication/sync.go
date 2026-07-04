package replication

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"time"
)

// SyncOptions configures a sync run.
type SyncOptions struct {
	// FlushInterval bounds how long a partially-filled batch may wait
	// before being flushed and checkpointed. Default 1s.
	FlushInterval time.Duration
	// Logger receives progress and error logs. Defaults to slog.Default().
	Logger *slog.Logger
}

func (o SyncOptions) withDefaults() SyncOptions {
	if o.FlushInterval <= 0 {
		o.FlushInterval = time.Second
	}
	if o.Logger == nil {
		o.Logger = slog.Default()
	}
	return o
}

// LSNAcker is implemented by sources that can acknowledge durably applied
// positions upstream (PgSource reports them to PostgreSQL so WAL can be
// recycled). Acking happens only after a successful flush AND checkpoint.
type LSNAcker interface {
	AckLSN(lsn uint64)
}

// Sync drives the pipeline: source -> transformer -> writer, checkpointing
// the applied LSN after every successful flush. It returns nil when the
// source is exhausted (io.EOF), and the first fatal error otherwise —
// including ErrCircuitOpen, which signals that the engine is persistently
// unavailable and consumption stopped without acking further LSNs.
func Sync(ctx context.Context, src EventSource, tr *Transformer, w *Writer, cp Checkpoint, opts SyncOptions) error {
	opts = opts.withDefaults()
	lastFlush := time.Now()
	checkpointed := uint64(0)
	acker, _ := src.(LSNAcker)

	flushAndCheckpoint := func() error {
		if err := w.Flush(ctx); err != nil {
			return err
		}
		lastFlush = time.Now()
		if lsn := w.AppliedLSN(); lsn > checkpointed {
			if err := cp.Save(lsn); err != nil {
				return fmt.Errorf("replication: checkpoint LSN %d: %w", lsn, err)
			}
			checkpointed = lsn
			if acker != nil {
				acker.AckLSN(lsn)
			}
		}
		return nil
	}

	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		// Bound Next by the flush interval: a live replication source blocks
		// indefinitely while the stream is idle, and buffered ops must still
		// be flushed and checkpointed on time.
		nextCtx, cancel := context.WithTimeout(ctx, opts.FlushInterval)
		ev, err := src.Next(nextCtx)
		cancel()
		if errors.Is(err, io.EOF) {
			return flushAndCheckpoint()
		}
		if err != nil {
			if ctx.Err() == nil && errors.Is(err, context.DeadlineExceeded) {
				// Source idle: flush whatever is buffered.
				if ferr := flushAndCheckpoint(); ferr != nil {
					return ferr
				}
				continue
			}
			return fmt.Errorf("replication: source: %w", err)
		}

		op, ok, err := tr.Transform(ev)
		if err != nil {
			// A malformed row is a poison event: log and skip rather than
			// wedging replication. Reconciliation repairs any divergence.
			opts.Logger.Error("skipping malformed change event",
				"table", ev.Table, "op", ev.Op.String(), "lsn", ev.LSN, "err", err)
			continue
		}
		if ok {
			if err := w.Apply(ctx, op); err != nil {
				return err
			}
		}
		if time.Since(lastFlush) >= opts.FlushInterval {
			if err := flushAndCheckpoint(); err != nil {
				return err
			}
		}
	}
}
