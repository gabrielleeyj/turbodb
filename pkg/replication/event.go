// Package replication implements the CDC pipeline that keeps a turbodb-engine
// in sync with PostgreSQL as the source of truth (SCOPE Component 7).
//
// The pipeline is: EventSource (logical replication) -> Transformer
// (table/column mapping + filters) -> Writer (batched engine RPCs with
// backpressure). The consumed position is persisted via Checkpoint so
// restarts resume from the last-committed LSN.
package replication

import "context"

// Op identifies the kind of change captured from PostgreSQL.
type Op int

// Change operations, mirroring logical replication message types.
const (
	OpInsert Op = iota + 1
	OpUpdate
	OpDelete
)

// String returns the lowercase name of the operation.
func (o Op) String() string {
	switch o {
	case OpInsert:
		return "insert"
	case OpUpdate:
		return "update"
	case OpDelete:
		return "delete"
	default:
		return "unknown"
	}
}

// ChangeEvent is one row-level change decoded from the replication stream.
type ChangeEvent struct {
	// Op is the change kind.
	Op Op
	// Table is the schema-qualified source table, e.g. "public.documents".
	Table string
	// LSN is the PostgreSQL WAL position of the change. Checkpointing this
	// LSN acknowledges everything up to and including the event.
	LSN uint64
	// Row holds column name -> value. For inserts and updates it is the new
	// row image; for deletes it holds at least the replica-identity columns.
	Row map[string]any
}

// EventSource produces an ordered stream of change events. The pglogrepl
// consumer implements this against a live PostgreSQL replication slot; tests
// use in-memory fakes.
type EventSource interface {
	// Next blocks until the next event is available, the source is
	// exhausted (io.EOF), or ctx is done.
	Next(ctx context.Context) (ChangeEvent, error)
	// Close releases the source's resources.
	Close() error
}
