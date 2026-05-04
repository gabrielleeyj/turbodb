package engine

import (
	"errors"
	"fmt"

	"github.com/gabrielleeyj/turbodb/pkg/index"
	"github.com/gabrielleeyj/turbodb/pkg/wal"
)

// replayWAL applies records from the WAL to the in-memory collections.
// MVP behavior: replay every record (no checkpoint pruning yet) since sealed
// segments are not persisted independently of the in-memory index.
func (e *Engine) replayWAL() error {
	dir := walDir(e.cfg.DataDir)

	var applied int
	err := wal.Iterate(dir, wal.IterateOptions{}, func(rec wal.Record) error {
		switch rec.Type {
		case wal.OpInsert:
			p, err := wal.DecodeInsert(rec.Payload)
			if err != nil {
				return fmt.Errorf("decode insert lsn=%d: %w", rec.LSN, err)
			}
			state, ok := e.collections[p.Collection]
			if !ok {
				e.logger.Warn("wal replay: insert for missing collection",
					"collection", p.Collection, "lsn", rec.LSN)
				return nil
			}
			err = state.coll.Insert(index.VectorEntry{
				ID:       p.ID,
				Values:   p.Values,
				Metadata: p.Metadata,
			})
			if err != nil {
				// Duplicate inserts can occur if recovery overlaps with prior partial state.
				e.logger.Warn("wal replay: insert error",
					"collection", p.Collection, "id", p.ID, "lsn", rec.LSN, "error", err)
				return nil
			}
			applied++

		case wal.OpDelete:
			p, err := wal.DecodeDelete(rec.Payload)
			if err != nil {
				return fmt.Errorf("decode delete lsn=%d: %w", rec.LSN, err)
			}
			state, ok := e.collections[p.Collection]
			if !ok {
				return nil
			}
			if err := state.coll.Delete(p.ID); err != nil && !errors.Is(err, errNotFound) {
				e.logger.Warn("wal replay: delete error",
					"collection", p.Collection, "id", p.ID, "lsn", rec.LSN, "error", err)
			}
			applied++

		case wal.OpSegmentSealed, wal.OpCheckpoint:
			// MVP: not load-bearing yet. Sealed segments live in memory; checkpoints
			// are advisory until we wire up sealed segment persistence.

		default:
			e.logger.Warn("wal replay: unknown record type", "type", rec.Type, "lsn", rec.LSN)
		}
		return nil
	})
	if err != nil {
		return err
	}

	if applied > 0 {
		e.logger.Info("engine: wal replay complete", "records_applied", applied)
	}
	return nil
}

// errNotFound is a sentinel for "vector not found" delete errors during replay.
var errNotFound = errors.New("not found")
