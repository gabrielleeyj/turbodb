package engine

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gabrielleeyj/turbodb/pkg/codebook"
	"github.com/gabrielleeyj/turbodb/pkg/index"
	"github.com/gabrielleeyj/turbodb/pkg/memory"
	"github.com/gabrielleeyj/turbodb/pkg/rotation"
	"github.com/gabrielleeyj/turbodb/pkg/search"
	"github.com/gabrielleeyj/turbodb/pkg/telemetry"
	"github.com/gabrielleeyj/turbodb/pkg/wal"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

// ErrCollectionNotFound is returned when a referenced collection does not exist.
var ErrCollectionNotFound = errors.New("engine: collection not found")

// ErrCollectionExists is returned when CreateCollection is called for a name
// that already exists.
var ErrCollectionExists = errors.New("engine: collection already exists")

// Engine coordinates collections, the write-ahead log, and persisted state.
// It is safe for concurrent use.
type Engine struct {
	cfg     EngineConfig
	logger  *slog.Logger
	metrics atomic.Pointer[telemetry.Metrics]
	tracer  trace.Tracer

	wal    *wal.WAL
	budget *memory.Budget

	mu          sync.RWMutex
	collections map[string]*collectionState
	closed      bool

	// segmentsSealedTotal counts seal completions over the engine's lifetime.
	// Sampled by the Prometheus stats source.
	segmentsSealedTotal atomic.Uint64
}

// collectionState bundles a Collection with the components needed to recreate
// or recover it.
type collectionState struct {
	config  CollectionConfig
	coll    *index.Collection
	planner *search.Planner
	rot     rotation.Rotator
	cb      *codebook.Codebook
}

// New opens or creates an Engine rooted at cfg.DataDir.
//
// Recovery flow:
//  1. Load all collection configs.
//  2. Instantiate empty Collection objects with matching rotator + codebook.
//  3. Replay the WAL onto them in LSN order.
//  4. Open the WAL for new appends.
func New(cfg EngineConfig) (*Engine, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}
	if err := os.MkdirAll(cfg.DataDir, 0o755); err != nil {
		return nil, fmt.Errorf("engine: create data dir: %w", err)
	}
	if err := os.MkdirAll(segmentsDir(cfg.DataDir), 0o755); err != nil {
		return nil, fmt.Errorf("engine: create segments dir: %w", err)
	}

	e := &Engine{
		cfg:         cfg,
		logger:      cfg.Logger,
		tracer:      telemetry.Tracer(),
		budget:      memory.NewBudget(cfg.MemoryBudgetBytes),
		collections: make(map[string]*collectionState),
	}

	configs, err := loadCollectionConfigs(cfg.DataDir)
	if err != nil {
		return nil, err
	}
	for _, cc := range configs {
		state, err := e.buildCollectionState(cc)
		if err != nil {
			return nil, fmt.Errorf("engine: rebuild collection %q: %w", cc.Name, err)
		}
		e.collections[cc.Name] = state
	}

	if err := e.replayWAL(); err != nil {
		return nil, fmt.Errorf("engine: wal replay: %w", err)
	}

	w, err := wal.Open(wal.Config{
		Dir:           walDir(cfg.DataDir),
		Logger:        cfg.Logger,
		FsyncObserver: e.observeWALFsync,
	})
	if err != nil {
		return nil, fmt.Errorf("engine: open wal: %w", err)
	}
	e.wal = w

	cfg.Logger.Info("engine: ready",
		"data_dir", cfg.DataDir,
		"collections", len(e.collections),
		"next_lsn", w.NextLSN(),
	)
	return e, nil
}

// Close releases all resources. Pending writes are flushed and fsynced.
func (e *Engine) Close() error {
	e.mu.Lock()
	if e.closed {
		e.mu.Unlock()
		return nil
	}
	e.closed = true
	collections := e.collections
	e.collections = nil
	e.mu.Unlock()

	var firstErr error
	for _, s := range collections {
		if err := s.coll.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if e.wal != nil {
		if err := e.wal.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// CreateCollection registers a new collection.
func (e *Engine) CreateCollection(ctx context.Context, cfg CollectionConfig) error {
	if err := cfg.Validate(); err != nil {
		return err
	}

	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return fmt.Errorf("engine: closed")
	}
	if _, ok := e.collections[cfg.Name]; ok {
		return fmt.Errorf("%w: %q", ErrCollectionExists, cfg.Name)
	}

	state, err := e.buildCollectionState(cfg)
	if err != nil {
		return err
	}

	if err := saveCollectionConfig(e.cfg.DataDir, cfg); err != nil {
		_ = state.coll.Close()
		return err
	}

	e.collections[cfg.Name] = state
	e.logger.Info("engine: collection created",
		"name", cfg.Name,
		"dim", cfg.Dim,
		"bit_width", cfg.BitWidth,
	)
	return nil
}

// DropCollection removes a collection and its in-memory state. Sealed segment
// files are not deleted by this MVP — they remain on disk for forensic access.
func (e *Engine) DropCollection(ctx context.Context, name string) error {
	e.mu.Lock()
	state, ok := e.collections[name]
	if !ok {
		e.mu.Unlock()
		return fmt.Errorf("%w: %q", ErrCollectionNotFound, name)
	}
	delete(e.collections, name)
	e.mu.Unlock()

	if err := state.coll.Close(); err != nil {
		e.logger.Warn("engine: drop collection close err", "name", name, "error", err)
	}
	if err := removeCollectionConfig(e.cfg.DataDir, name); err != nil {
		return err
	}
	e.logger.Info("engine: collection dropped", "name", name)
	return nil
}

// ListCollections returns a snapshot of all collection configs.
func (e *Engine) ListCollections() []CollectionConfig {
	e.mu.RLock()
	defer e.mu.RUnlock()
	out := make([]CollectionConfig, 0, len(e.collections))
	for _, s := range e.collections {
		out = append(out, s.config)
	}
	return out
}

// DescribeCollection returns the config and stats for one collection.
func (e *Engine) DescribeCollection(name string) (CollectionConfig, index.CollectionStats, error) {
	e.mu.RLock()
	state, ok := e.collections[name]
	e.mu.RUnlock()
	if !ok {
		return CollectionConfig{}, index.CollectionStats{}, fmt.Errorf("%w: %q", ErrCollectionNotFound, name)
	}
	return state.config, state.coll.Stats(), nil
}

// Insert appends a vector to the named collection. The operation is durably
// logged before applying to the in-memory index.
func (e *Engine) Insert(ctx context.Context, collection string, entry index.VectorEntry) error {
	if err := validateVector(entry, 0); err != nil {
		return err
	}

	e.mu.RLock()
	state, ok := e.collections[collection]
	e.mu.RUnlock()
	if !ok {
		return fmt.Errorf("%w: %q", ErrCollectionNotFound, collection)
	}
	if len(entry.Values) != state.config.Dim {
		return fmt.Errorf("engine: insert: expected dim %d, got %d", state.config.Dim, len(entry.Values))
	}

	payload, err := wal.EncodeInsert(wal.InsertPayload{
		Collection: collection,
		ID:         entry.ID,
		Values:     entry.Values,
		Metadata:   entry.Metadata,
	})
	if err != nil {
		return err
	}
	if _, err := e.wal.Append(wal.OpInsert, payload); err != nil {
		return fmt.Errorf("engine: wal append: %w", err)
	}
	if err := state.coll.Insert(entry); err != nil {
		return err
	}
	e.metrics.Load().AddInserts(1)
	return nil
}

// Delete tombstones a vector from the named collection.
func (e *Engine) Delete(ctx context.Context, collection, id string) error {
	if id == "" {
		return fmt.Errorf("engine: delete: id must not be empty")
	}

	e.mu.RLock()
	state, ok := e.collections[collection]
	e.mu.RUnlock()
	if !ok {
		return fmt.Errorf("%w: %q", ErrCollectionNotFound, collection)
	}

	payload, err := wal.EncodeDelete(wal.DeletePayload{Collection: collection, ID: id})
	if err != nil {
		return err
	}
	if _, err := e.wal.Append(wal.OpDelete, payload); err != nil {
		return fmt.Errorf("engine: wal append: %w", err)
	}
	return state.coll.Delete(id)
}

// Search returns the top-K most similar vectors to query along with a Plan
// describing how the planner executed.
func (e *Engine) Search(ctx context.Context, collection string, query []float32, opts search.Options) ([]index.SearchResult, search.Plan, error) {
	ctx, span := e.tracer.Start(ctx, "engine.Search",
		trace.WithAttributes(
			attribute.String("collection", collection),
			attribute.Int("top_k", opts.TopK),
			attribute.Bool("rerank", opts.Rerank),
			attribute.Bool("exact", opts.Exact),
		),
	)
	defer span.End()
	start := time.Now()

	if err := validateValues(query); err != nil {
		span.SetStatus(codes.Error, err.Error())
		return nil, search.Plan{}, fmt.Errorf("engine: search: %w", err)
	}

	e.mu.RLock()
	state, ok := e.collections[collection]
	e.mu.RUnlock()
	if !ok {
		err := fmt.Errorf("%w: %q", ErrCollectionNotFound, collection)
		span.SetStatus(codes.Error, "collection not found")
		return nil, search.Plan{}, err
	}

	results, plan, err := state.planner.Run(ctx, query, opts)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
		return results, plan, err
	}
	span.SetAttributes(
		attribute.Int("plan.segments_searched", plan.SegmentsSearched),
		attribute.Int("plan.candidates_considered", plan.CandidatesConsidered),
		attribute.Int("plan.effective_top_k", plan.EffectiveTopK),
		attribute.Bool("plan.reranked", plan.Reranked),
	)
	e.metrics.Load().ObserveSearchLatency(time.Since(start).Seconds())
	return results, plan, nil
}

// Flush forces sealing of the active growing segment in the named collection.
func (e *Engine) Flush(ctx context.Context, collection string) error {
	e.mu.RLock()
	state, ok := e.collections[collection]
	e.mu.RUnlock()
	if !ok {
		return fmt.Errorf("%w: %q", ErrCollectionNotFound, collection)
	}
	return state.coll.Flush()
}

// Stats returns runtime statistics for a collection.
func (e *Engine) Stats(collection string) (index.CollectionStats, error) {
	e.mu.RLock()
	state, ok := e.collections[collection]
	e.mu.RUnlock()
	if !ok {
		return index.CollectionStats{}, fmt.Errorf("%w: %q", ErrCollectionNotFound, collection)
	}
	return state.coll.Stats(), nil
}

// MemoryStats reports the engine-wide memory budget headline figures. A
// Capacity of zero indicates the budget is unlimited (accounting only).
type MemoryStats struct {
	UsedBytes     int64
	CapacityBytes int64
	Unlimited     bool
}

// MemoryStats returns the engine-wide memory budget snapshot.
func (e *Engine) MemoryStats() MemoryStats {
	return MemoryStats{
		UsedBytes:     e.budget.Used(),
		CapacityBytes: e.budget.Capacity(),
		Unlimited:     e.budget.Unlimited(),
	}
}

// buildCollectionState constructs an in-memory collection from a config.
func (e *Engine) buildCollectionState(cfg CollectionConfig) (*collectionState, error) {
	rot, err := rotation.NewHadamardRotator(cfg.Dim, cfg.RotatorSeed)
	if err != nil {
		return nil, fmt.Errorf("engine: rotator: %w", err)
	}

	cb, err := codebook.Load(cfg.Dim, cfg.BitWidth)
	if err != nil {
		return nil, fmt.Errorf("engine: codebook (dim=%d, bw=%d): %w", cfg.Dim, cfg.BitWidth, err)
	}

	coll, err := index.NewCollection(index.CollectionConfig{
		Name:          cfg.Name,
		Dim:           cfg.Dim,
		BitWidth:      cfg.BitWidth,
		Rotator:       rot,
		Codebook:      cb,
		SealThreshold: e.cfg.SealThreshold,
		DataDir:       segmentsDir(e.cfg.DataDir),
		Logger:        e.logger.With("collection", cfg.Name),
		Budget:        e.budget,
		OnSealed: func(_ string, _ int, _ int64) {
			e.segmentsSealedTotal.Add(1)
			e.metrics.Load().IncSegmentsSealed()
		},
	})
	if err != nil {
		return nil, fmt.Errorf("engine: collection: %w", err)
	}

	planner, err := search.NewPlanner(coll, nil)
	if err != nil {
		_ = coll.Close()
		return nil, fmt.Errorf("engine: planner: %w", err)
	}

	return &collectionState{config: cfg, coll: coll, planner: planner, rot: rot, cb: cb}, nil
}

// validateValues checks that a vector contains no NaN/Inf entries.
func validateValues(values []float32) error {
	if len(values) == 0 {
		return fmt.Errorf("values must not be empty")
	}
	for i, v := range values {
		f := float64(v)
		if math.IsNaN(f) {
			return fmt.Errorf("values[%d] is NaN", i)
		}
		if math.IsInf(f, 0) {
			return fmt.Errorf("values[%d] is Inf", i)
		}
	}
	return nil
}

// SegmentsActive reports the total growing+sealed segments across all
// collections; satisfies telemetry.StatsSource.
func (e *Engine) SegmentsActive() int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	total := 0
	for _, s := range e.collections {
		st := s.coll.Stats()
		total += st.GrowingSegmentCount + st.SealedSegmentCount
	}
	return total
}

// SegmentsSealedTotal reports the lifetime number of seal completions.
// Satisfies telemetry.StatsSource.
func (e *Engine) SegmentsSealedTotal() uint64 {
	return e.segmentsSealedTotal.Load()
}

// HostMemoryBytes reports total host bytes pinned by sealed segments.
// Satisfies telemetry.StatsSource.
func (e *Engine) HostMemoryBytes() int64 {
	return e.budget.Used()
}

// GPUMemoryBytes reports GPU bytes pinned. The CPU MVP returns zero;
// kept here so the gauge wires up cleanly when the GPU path lands.
func (e *Engine) GPUMemoryBytes() int64 { return 0 }

// AttachMetrics installs the Prometheus metrics bundle. Safe to call
// after construction so the engine can be passed as the StatsSource
// when New(opts) is invoked. Passing nil detaches metrics.
func (e *Engine) AttachMetrics(m *telemetry.Metrics) {
	e.metrics.Store(m)
}

// observeWALFsync forwards each fsync latency to the attached Metrics.
// Reads the atomic pointer so updates from AttachMetrics take effect
// immediately without restarting the WAL.
func (e *Engine) observeWALFsync(d time.Duration) {
	e.metrics.Load().ObserveWALFsyncLatency(d.Seconds())
}

// validateVector checks the basic shape of a VectorEntry.
// expectedDim is enforced when > 0.
func validateVector(entry index.VectorEntry, expectedDim int) error {
	if entry.ID == "" {
		return fmt.Errorf("engine: vector id must not be empty")
	}
	if err := validateValues(entry.Values); err != nil {
		return fmt.Errorf("engine: vector %q: %w", entry.ID, err)
	}
	if expectedDim > 0 && len(entry.Values) != expectedDim {
		return fmt.Errorf("engine: vector %q: expected dim %d, got %d", entry.ID, expectedDim, len(entry.Values))
	}
	return nil
}
