package replication

import (
	"context"
	"errors"
	"fmt"
	"math/rand/v2"
	"time"
)

// VectorRecord is one vector destined for the engine.
type VectorRecord struct {
	ID     string
	Vector []float32
}

// EngineClient is the write surface the sync pipeline needs from the engine.
// The gRPC client implements this; tests use fakes.
type EngineClient interface {
	InsertBatch(ctx context.Context, collection string, records []VectorRecord) error
	DeleteBatch(ctx context.Context, collection string, ids []string) error
}

// ErrCircuitOpen is returned once the writer has seen too many consecutive
// flush failures. The caller must stop consuming (and stop acking LSNs) and
// surface a critical alert; a new Writer is required to resume.
var ErrCircuitOpen = errors.New("replication: circuit breaker open")

// WriterConfig tunes batching and failure handling.
type WriterConfig struct {
	// MaxBatch is the number of buffered ops that triggers an automatic
	// flush. Default 256.
	MaxBatch int
	// MaxRetries is how many times a failed flush is retried with
	// exponential backoff before counting as a failure. Default 4.
	MaxRetries int
	// BaseBackoff is the initial retry delay; each retry doubles it and
	// adds up to 50% jitter. Default 100ms.
	BaseBackoff time.Duration
	// BreakerThreshold is the number of consecutive failed flushes (each
	// already retried MaxRetries times) that opens the circuit. Default 3.
	BreakerThreshold int
	// Sleep is injectable for tests; defaults to time.Sleep via ctx-aware
	// waiting.
	Sleep func(ctx context.Context, d time.Duration) error
}

func (c WriterConfig) withDefaults() WriterConfig {
	if c.MaxBatch <= 0 {
		c.MaxBatch = 256
	}
	if c.MaxRetries <= 0 {
		c.MaxRetries = 4
	}
	if c.BaseBackoff <= 0 {
		c.BaseBackoff = 100 * time.Millisecond
	}
	if c.BreakerThreshold <= 0 {
		c.BreakerThreshold = 3
	}
	if c.Sleep == nil {
		c.Sleep = sleepCtx
	}
	return c
}

func sleepCtx(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// batch is a run of consecutive ops sharing kind and collection, so ordering
// across kinds (e.g. delete-then-insert of the same id) is preserved.
type batch struct {
	kind       EngineOpKind
	collection string
	records    []VectorRecord
	ids        []string
	maxLSN     uint64
}

// Writer buffers engine ops and flushes them as batched RPCs with retry,
// exponential backoff with jitter, and a consecutive-failure circuit breaker.
// It is not safe for concurrent use; the sync loop is single-threaded by
// design (source order must be preserved).
type Writer struct {
	client EngineClient
	cfg    WriterConfig

	pending     []batch
	pendingOps  int
	appliedLSN  uint64
	consecFails int
	circuitOpen bool
}

// NewWriter creates a Writer for the given engine client.
func NewWriter(client EngineClient, cfg WriterConfig) *Writer {
	return &Writer{client: client, cfg: cfg.withDefaults()}
}

// AppliedLSN returns the highest LSN whose op has been durably flushed to
// the engine. It is safe to checkpoint.
func (w *Writer) AppliedLSN() uint64 { return w.appliedLSN }

// Apply buffers an op, flushing automatically when the batch limit is hit.
func (w *Writer) Apply(ctx context.Context, op EngineOp) error {
	if w.circuitOpen {
		return ErrCircuitOpen
	}
	if op.Kind != EngineUpsert && op.Kind != EngineDelete {
		return fmt.Errorf("replication: unknown engine op kind %d", op.Kind)
	}

	w.appendOp(op)
	if w.pendingOps >= w.cfg.MaxBatch {
		return w.Flush(ctx)
	}
	return nil
}

func (w *Writer) appendOp(op EngineOp) {
	n := len(w.pending)
	if n == 0 || w.pending[n-1].kind != op.Kind || w.pending[n-1].collection != op.Collection {
		w.pending = append(w.pending, batch{kind: op.Kind, collection: op.Collection})
		n++
	}
	b := &w.pending[n-1]
	switch op.Kind {
	case EngineUpsert:
		b.records = append(b.records, VectorRecord{ID: op.ID, Vector: op.Vector})
	case EngineDelete:
		b.ids = append(b.ids, op.ID)
	}
	if op.LSN > b.maxLSN {
		b.maxLSN = op.LSN
	}
	w.pendingOps++
}

// Flush sends all buffered batches in order. On success the applied LSN
// advances; after MaxRetries failed attempts the failure counts toward the
// circuit breaker and the buffered ops are retained for a later retry.
func (w *Writer) Flush(ctx context.Context) error {
	if w.circuitOpen {
		return ErrCircuitOpen
	}
	for len(w.pending) > 0 {
		b := w.pending[0]
		if err := w.sendWithRetry(ctx, b); err != nil {
			w.consecFails++
			if w.consecFails >= w.cfg.BreakerThreshold {
				w.circuitOpen = true
				return fmt.Errorf("%w: %d consecutive flush failures, last: %v",
					ErrCircuitOpen, w.consecFails, err)
			}
			return fmt.Errorf("replication: flush %s %s: %w", opKindName(b.kind), b.collection, err)
		}
		w.consecFails = 0
		if b.maxLSN > w.appliedLSN {
			w.appliedLSN = b.maxLSN
		}
		w.pending = w.pending[1:]
		w.pendingOps -= len(b.records) + len(b.ids)
	}
	return nil
}

func (w *Writer) sendWithRetry(ctx context.Context, b batch) error {
	var lastErr error
	backoff := w.cfg.BaseBackoff
	for attempt := 0; attempt <= w.cfg.MaxRetries; attempt++ {
		if attempt > 0 {
			jitter := time.Duration(rand.Int64N(int64(backoff)/2 + 1)) // #nosec G404 -- jitter, not cryptographic
			if err := w.cfg.Sleep(ctx, backoff+jitter); err != nil {
				return err
			}
			backoff *= 2
		}
		lastErr = w.send(ctx, b)
		if lastErr == nil {
			return nil
		}
		if ctx.Err() != nil {
			return lastErr
		}
	}
	return lastErr
}

func (w *Writer) send(ctx context.Context, b batch) error {
	switch b.kind {
	case EngineUpsert:
		return w.client.InsertBatch(ctx, b.collection, b.records)
	case EngineDelete:
		return w.client.DeleteBatch(ctx, b.collection, b.ids)
	default:
		return fmt.Errorf("unknown batch kind %d", b.kind)
	}
}

func opKindName(k EngineOpKind) string {
	switch k {
	case EngineUpsert:
		return "upsert"
	case EngineDelete:
		return "delete"
	default:
		return "unknown"
	}
}
