// Package pgipc adapts the pgproto IPC protocol onto the TurboDB engine. It is
// the engine-side bridge for the pg_turboquant PostgreSQL extension: it
// translates protocol messages into engine operations and maps PostgreSQL item
// pointers (uint64 TIDs) to engine vector IDs.
package pgipc

import (
	"context"
	"errors"
	"fmt"
	"strconv"

	"github.com/gabrielleeyj/turbodb/internal/engine"
	"github.com/gabrielleeyj/turbodb/internal/pgproto"
	"github.com/gabrielleeyj/turbodb/pkg/index"
	"github.com/gabrielleeyj/turbodb/pkg/search"
)

// Adapter implements pgproto.Handler backed by an *engine.Engine.
type Adapter struct {
	eng *engine.Engine
}

// NewAdapter constructs an Adapter over eng.
func NewAdapter(eng *engine.Engine) *Adapter {
	return &Adapter{eng: eng}
}

// tidToID renders a PostgreSQL TID as a stable engine vector ID.
func tidToID(tid uint64) string { return strconv.FormatUint(tid, 10) }

// idToTID parses an engine vector ID back into a TID. IDs created by this
// adapter are decimal TIDs; foreign IDs hash to 0.
func idToTID(id string) uint64 {
	v, err := strconv.ParseUint(id, 10, 64)
	if err != nil {
		return 0
	}
	return v
}

// BuildBegin creates the collection if it does not already exist. A repeated
// build on an existing collection is treated as idempotent.
func (a *Adapter) BuildBegin(ctx context.Context, m pgproto.BuildBegin) error {
	cfg := engine.CollectionConfig{
		Name:        m.Collection,
		Dim:         int(m.Dim),
		BitWidth:    int(m.BitWidth),
		Metric:      engine.MetricInnerProduct,
		Variant:     engine.VariantMSE,
		RotatorSeed: m.RotatorSeed,
	}
	if err := a.eng.CreateCollection(ctx, cfg); err != nil {
		if errors.Is(err, engine.ErrCollectionExists) {
			return nil
		}
		return fmt.Errorf("pgipc: build begin: %w", err)
	}
	return nil
}

// Insert adds a vector under the TID-derived ID.
func (a *Adapter) Insert(ctx context.Context, collection string, m pgproto.VectorMsg) error {
	return a.eng.Insert(ctx, collection, index.VectorEntry{
		ID:     tidToID(m.TID),
		Values: m.Values,
	})
}

// Delete tombstones the TID-derived ID.
func (a *Adapter) Delete(ctx context.Context, collection string, m pgproto.DeleteMsg) error {
	return a.eng.Delete(ctx, collection, tidToID(m.TID))
}

// Commit flushes (seals) the collection to finalize a build.
func (a *Adapter) Commit(ctx context.Context, collection string) error {
	return a.eng.Flush(ctx, collection)
}

// Search runs a query and converts engine results to protocol rows.
func (a *Adapter) Search(ctx context.Context, m pgproto.SearchBegin) ([]pgproto.ResultMsg, error) {
	opts := search.Options{
		TopK:             int(m.TopK),
		OversearchFactor: float64(m.OversearchFactor),
		Rerank:           m.Rerank,
		Exact:            m.Exact,
	}
	if opts.TopK < 1 {
		opts.TopK = 1
	}
	results, _, err := a.eng.Search(ctx, m.Collection, m.Query, opts)
	if err != nil {
		return nil, fmt.Errorf("pgipc: search: %w", err)
	}
	rows := make([]pgproto.ResultMsg, len(results))
	for i, r := range results {
		rows[i] = pgproto.ResultMsg{TID: idToTID(r.ID), Score: r.Score}
	}
	return rows, nil
}

// Stats returns collection statistics.
func (a *Adapter) Stats(_ context.Context, collection string) (pgproto.StatsReply, error) {
	stats, err := a.eng.Stats(collection)
	if err != nil {
		return pgproto.StatsReply{}, fmt.Errorf("pgipc: stats: %w", err)
	}
	return pgproto.StatsReply{
		VectorCount:    uint64(stats.VectorCount),         // #nosec G115 -- counts are non-negative
		SealedSegments: uint32(stats.SealedSegmentCount),  // #nosec G115 -- segment counts are far below uint32 max
		GrowingSegment: uint32(stats.GrowingSegmentCount), // #nosec G115 -- segment counts are far below uint32 max
		PinnedBytes:    uint64(stats.PinnedBytes),         // #nosec G115 -- counts are non-negative
	}, nil
}
