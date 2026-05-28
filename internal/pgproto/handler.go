package pgproto

import "context"

// Handler implements the engine-side semantics of the IPC protocol. A single
// Handler is shared across all connections; per-connection state (current
// collection, search cursor) is managed by the server.
//
// Collection routing: a backend connection operates on one collection at a
// time. BuildBegin (index build) and SearchBegin (query) both name the
// collection; Insert/Delete/BuildVector/BuildCommit/Stats act on the
// connection's current collection established by the most recent BuildBegin or
// SearchBegin.
type Handler interface {
	// BuildBegin creates (or opens) a collection for bulk index build.
	BuildBegin(ctx context.Context, m BuildBegin) error
	// Insert adds a single vector to a collection (also used for BUILD_VECTOR).
	Insert(ctx context.Context, collection string, m VectorMsg) error
	// Delete tombstones a tid.
	Delete(ctx context.Context, collection string, m DeleteMsg) error
	// Commit finalizes a build (flush/seal). Used for BUILD_COMMIT.
	Commit(ctx context.Context, collection string) error
	// Search runs a query and returns the full ordered result set, which the
	// server streams back one row at a time via SEARCH_NEXT.
	Search(ctx context.Context, m SearchBegin) ([]ResultMsg, error)
	// Stats returns collection statistics.
	Stats(ctx context.Context, collection string) (StatsReply, error)
}
