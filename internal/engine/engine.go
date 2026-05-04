package engine

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"os"
	"sync"

	"github.com/gabrielleeyj/turbodb/pkg/codebook"
	"github.com/gabrielleeyj/turbodb/pkg/index"
	"github.com/gabrielleeyj/turbodb/pkg/rotation"
	"github.com/gabrielleeyj/turbodb/pkg/wal"
)

// ErrCollectionNotFound is returned when a referenced collection does not exist.
var ErrCollectionNotFound = errors.New("engine: collection not found")

// ErrCollectionExists is returned when CreateCollection is called for a name
// that already exists.
var ErrCollectionExists = errors.New("engine: collection already exists")

// Engine coordinates collections, the write-ahead log, and persisted state.
// It is safe for concurrent use.
type Engine struct {
	cfg    EngineConfig
	logger *slog.Logger

	wal *wal.WAL

	mu          sync.RWMutex
	collections map[string]*collectionState
	closed      bool
}

// collectionState bundles a Collection with the components needed to recreate
// or recover it.
type collectionState struct {
	config CollectionConfig
	coll   *index.Collection
	rot    rotation.Rotator
	cb     *codebook.Codebook
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
		Dir:    walDir(cfg.DataDir),
		Logger: cfg.Logger,
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
	return state.coll.Insert(entry)
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

// Search returns the top-K most similar vectors to query.
func (e *Engine) Search(ctx context.Context, collection string, query []float32, topK int) ([]index.SearchResult, error) {
	if topK < 1 || topK > 1000 {
		return nil, fmt.Errorf("engine: search: top_k must be 1..1000, got %d", topK)
	}
	if err := validateValues(query); err != nil {
		return nil, fmt.Errorf("engine: search: %w", err)
	}

	e.mu.RLock()
	state, ok := e.collections[collection]
	e.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrCollectionNotFound, collection)
	}
	if len(query) != state.config.Dim {
		return nil, fmt.Errorf("engine: search: expected dim %d, got %d", state.config.Dim, len(query))
	}
	return state.coll.Search(query, topK)
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
	})
	if err != nil {
		return nil, fmt.Errorf("engine: collection: %w", err)
	}

	return &collectionState{config: cfg, coll: coll, rot: rot, cb: cb}, nil
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
