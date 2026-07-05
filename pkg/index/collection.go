package index

import (
	"context"
	"fmt"
	"log/slog"
	"sort"
	"sync"
	"time"

	"github.com/gabrielleeyj/turbodb/pkg/codebook"
	"github.com/gabrielleeyj/turbodb/pkg/memory"
	"github.com/gabrielleeyj/turbodb/pkg/rotation"
)

// DefaultSealThreshold is the default number of vectors that triggers segment sealing.
const DefaultSealThreshold = 1_000_000

// CollectionConfig holds configuration for a Collection.
type CollectionConfig struct {
	Name          string
	Dim           int
	BitWidth      int
	Rotator       rotation.Rotator
	Codebook      *codebook.Codebook
	SealThreshold int    // Vectors per growing segment before auto-seal.
	DataDir       string // Directory for persisted segment files.
	Logger        *slog.Logger
	// Budget is an optional engine-wide memory budget. When set, sealed
	// segments acquire their estimated byte cost on creation and release on
	// Close. Nil means unbounded.
	Budget *memory.Budget
	// OnSealed, when non-nil, is invoked after each successful seal — auto
	// or via Flush. Used for telemetry; must not block.
	OnSealed func(segmentID string, vectors int, pinnedBytes int64)
}

// Collection manages the lifecycle of segments for a single logical index.
// It routes inserts to the active growing segment, seals segments when they
// reach a size threshold, and merges search results across all segments.
type Collection struct {
	mu sync.RWMutex

	name     string
	dim      int
	bitWidth int
	rotator  rotation.Rotator
	cb       *codebook.Codebook

	growing    *GrowingSegment
	sealed     []*SealedSegment
	tombstones *TombstoneLog

	sealThreshold int
	dataDir       string
	segCounter    int
	logger        *slog.Logger

	budget        *memory.Budget
	pinnedBytes   int64 // bytes acquired from budget by this collection's sealed segments
	pinnedSegSize map[string]int64
	onSealed      func(segmentID string, vectors int, pinnedBytes int64)

	// Sealing coordination.
	sealCh    chan sealRequest
	pendingWg sync.WaitGroup // Tracks in-flight seal operations.
	cancelF   context.CancelFunc
	wg        sync.WaitGroup
}

type sealRequest struct {
	segment *GrowingSegment
	done    chan error
}

// NewCollection creates a new collection and starts the background sealer.
func NewCollection(cfg CollectionConfig) (*Collection, error) {
	if cfg.Name == "" {
		return nil, fmt.Errorf("collection: name must not be empty")
	}
	if cfg.Dim < 1 {
		return nil, fmt.Errorf("collection: dim must be >= 1, got %d", cfg.Dim)
	}
	if cfg.Rotator == nil {
		return nil, fmt.Errorf("collection: rotator must not be nil")
	}
	if cfg.Codebook == nil {
		return nil, fmt.Errorf("collection: codebook must not be nil")
	}
	if cfg.BitWidth < 1 || cfg.BitWidth > 8 {
		return nil, fmt.Errorf("collection: bitWidth must be 1..8, got %d", cfg.BitWidth)
	}

	threshold := cfg.SealThreshold
	if threshold <= 0 {
		threshold = DefaultSealThreshold
	}

	logger := cfg.Logger
	if logger == nil {
		logger = slog.Default()
	}

	c := &Collection{
		name:          cfg.Name,
		dim:           cfg.Dim,
		bitWidth:      cfg.BitWidth,
		rotator:       cfg.Rotator,
		cb:            cfg.Codebook,
		tombstones:    NewTombstoneLog(),
		sealThreshold: threshold,
		dataDir:       cfg.DataDir,
		logger:        logger,
		sealCh:        make(chan sealRequest, 4),
		budget:        cfg.Budget,
		pinnedSegSize: make(map[string]int64),
		onSealed:      cfg.OnSealed,
	}

	// Create initial growing segment.
	growing, err := c.newGrowingSegment()
	if err != nil {
		return nil, fmt.Errorf("collection: create initial segment: %w", err)
	}
	c.growing = growing

	// Start background sealer.
	ctx, cancel := context.WithCancel(context.Background())
	c.cancelF = cancel
	c.wg.Add(1)
	go c.sealerLoop(ctx)

	return c, nil
}

func (c *Collection) newGrowingSegment() (*GrowingSegment, error) {
	c.segCounter++
	id := fmt.Sprintf("%s-seg-%04d", c.name, c.segCounter)
	return NewGrowingSegment(GrowingSegmentConfig{
		ID:       id,
		Dim:      c.dim,
		Rotator:  c.rotator,
		Codebook: c.cb,
		BitWidth: c.bitWidth,
	})
}

// Name returns the collection name.
func (c *Collection) Name() string { return c.name }

// Dim returns the vector dimensionality.
func (c *Collection) Dim() int { return c.dim }

// Insert adds a vector to the active growing segment. If the segment reaches
// the seal threshold, it is sealed asynchronously and a new one is created.
func (c *Collection) Insert(entry VectorEntry) error {
	c.mu.Lock()

	// Check if deleted — re-insert after delete is allowed.
	c.tombstones.Remove(entry.ID)

	// Check for duplicates across all segments.
	if c.growing.Contains(entry.ID) {
		c.mu.Unlock()
		return fmt.Errorf("collection: duplicate vector ID %q", entry.ID)
	}
	for _, s := range c.sealed {
		if s.Contains(entry.ID) {
			c.mu.Unlock()
			return fmt.Errorf("collection: duplicate vector ID %q", entry.ID)
		}
	}

	err := c.growing.Insert(entry)
	if err != nil {
		c.mu.Unlock()
		return fmt.Errorf("collection insert: %w", err)
	}
	return c.maybeRotateLocked()
}

// Upsert inserts the vector or replaces any existing copy with the same ID.
// This is the idempotent write path required by at-least-once CDC delivery:
// redelivered transactions and row updates must not fail.
//
// A copy resident in a sealed segment cannot be rewritten in place (sealed
// segments hold immutable quantized codes), so it is masked with a tombstone
// and the new copy lives in the growing segment. See sealSegment for how the
// mask is resolved when that growing segment itself seals.
func (c *Collection) Upsert(entry VectorEntry) error {
	c.mu.Lock()

	sealedHas := false
	for _, s := range c.sealed {
		if s.Contains(entry.ID) {
			sealedHas = true
			break
		}
	}
	if sealedHas {
		// Mask the stale sealed copy. Growing-segment reads ignore
		// tombstones (deletes there are physical), so the new copy
		// stays visible.
		c.tombstones.Delete(entry.ID)
	} else {
		// Supersede any prior delete.
		c.tombstones.Remove(entry.ID)
	}

	if err := c.growing.Upsert(entry); err != nil {
		c.mu.Unlock()
		return fmt.Errorf("collection upsert: %w", err)
	}
	return c.maybeRotateLocked()
}

// maybeRotateLocked seals the growing segment once it reaches the threshold.
// Caller holds c.mu; the lock is released before returning.
func (c *Collection) maybeRotateLocked() error {
	if c.growing.Count() >= c.sealThreshold {
		old := c.growing
		newSeg, err := c.newGrowingSegment()
		if err != nil {
			c.mu.Unlock()
			return fmt.Errorf("collection: rotate segment: %w", err)
		}
		c.growing = newSeg
		c.pendingWg.Add(1)
		c.mu.Unlock()

		// Queue seal request (channel is buffered, sealer loop drains it).
		req := sealRequest{segment: old, done: make(chan error, 1)}
		c.sealCh <- req
		go func() {
			defer c.pendingWg.Done()
			if err := <-req.done; err != nil {
				c.logger.Error("background seal failed", "segment", old.ID(), "error", err)
			}
		}()
		return nil
	}

	c.mu.Unlock()
	return nil
}

// Delete marks a vector as deleted via tombstone. The vector is excluded from
// future search results.
func (c *Collection) Delete(id string) error {
	if id == "" {
		return fmt.Errorf("collection delete: ID must not be empty")
	}

	c.mu.RLock()
	removed := c.growing.Remove(id)
	sealedHas := false
	for _, s := range c.sealed {
		if s.Contains(id) {
			sealedHas = true
			break
		}
	}
	c.mu.RUnlock()

	if sealedHas {
		c.tombstones.Delete(id)
	}
	if !removed && !sealedHas {
		return fmt.Errorf("collection delete: vector ID %q not found", id)
	}
	return nil
}

// Search queries all segments and merges results into a single top-K list.
// Results are returned in descending score order.
func (c *Collection) Search(query []float32, topK int) ([]SearchResult, error) {
	if len(query) != c.dim {
		return nil, fmt.Errorf("collection search: expected dim %d, got %d", c.dim, len(query))
	}
	if topK < 1 {
		return nil, fmt.Errorf("collection search: topK must be >= 1")
	}

	c.mu.RLock()
	growing := c.growing
	sealed := make([]*SealedSegment, len(c.sealed))
	copy(sealed, c.sealed)
	c.mu.RUnlock()

	// Search all segments in parallel.
	type segResult struct {
		results []SearchResult
		err     error
	}

	totalSegments := 1 + len(sealed)
	ch := make(chan segResult, totalSegments)

	// Search growing segment. Growing deletes are physical, and a
	// tombstone may exist solely to mask a stale sealed copy of an
	// upserted id, so growing reads ignore the tombstone log.
	go func() {
		results, err := growing.Search(query, topK, nil)
		ch <- segResult{results, err}
	}()

	// Search sealed segments.
	for _, s := range sealed {
		go func() {
			results, err := s.Search(query, topK, c.tombstones)
			ch <- segResult{results, err}
		}()
	}

	// Merge results.
	var allResults []SearchResult
	for range totalSegments {
		sr := <-ch
		if sr.err != nil {
			return nil, fmt.Errorf("collection search: segment error: %w", sr.err)
		}
		allResults = append(allResults, sr.results...)
	}

	return mergeTopK(dedupByID(allResults), topK), nil
}

// Stats returns collection statistics.
func (c *Collection) Stats() CollectionStats {
	c.mu.RLock()
	growingCount := c.growing.Count()
	sealedCount := len(c.sealed)
	var totalVectors int
	totalVectors += growingCount
	for _, s := range c.sealed {
		totalVectors += s.Count()
	}
	pinnedBytes := c.pinnedBytes
	c.mu.RUnlock()

	return CollectionStats{
		VectorCount:         totalVectors,
		GrowingSegmentCount: 1,
		SealedSegmentCount:  sealedCount,
		TombstoneCount:      c.tombstones.Count(),
		PinnedBytes:         pinnedBytes,
	}
}

// Snapshot returns all live vectors across the growing and sealed segments.
// Growing-segment vectors are exact; sealed-segment vectors are reconstructed
// from their quantized codes and therefore lossy. Tombstoned ids are excluded.
// Intended for export and inspection.
func (c *Collection) Snapshot() ([]VectorEntry, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var out []VectorEntry
	// Growing deletes are physical; every growing entry is live.
	out = append(out, c.growing.Entries()...)
	for _, s := range c.sealed {
		entries, err := s.Reconstruct(c.tombstones)
		if err != nil {
			return nil, fmt.Errorf("collection snapshot: segment %s: %w", s.ID(), err)
		}
		out = append(out, entries...)
	}
	return out, nil
}

// IDs returns the ids of all live vectors (tombstoned ids excluded), sorted
// in bytewise ascending order. Unlike Snapshot it does not reconstruct
// vectors, so it is cheap enough for reconciliation scans.
func (c *Collection) IDs() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()

	seen := make(map[string]struct{})
	for _, entry := range c.growing.Entries() {
		seen[entry.ID] = struct{}{}
	}
	for _, s := range c.sealed {
		for _, id := range s.IDs() {
			if !c.tombstones.IsDeleted(id) {
				seen[id] = struct{}{}
			}
		}
	}
	out := make([]string, 0, len(seen))
	for id := range seen {
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}

// CollectionStats holds statistics about a collection.
type CollectionStats struct {
	VectorCount         int
	GrowingSegmentCount int
	SealedSegmentCount  int
	TombstoneCount      int
	// PinnedBytes is the byte cost currently held against the engine's
	// memory budget by this collection's sealed segments.
	PinnedBytes int64
}

// Flush forces sealing of the current growing segment (even if below threshold)
// and returns after the seal completes. Also waits for any pending auto-seals.
func (c *Collection) Flush() error {
	// Wait for any pending auto-seals to complete first.
	c.pendingWg.Wait()

	c.mu.Lock()
	if c.growing.Count() == 0 {
		c.mu.Unlock()
		return nil
	}

	old := c.growing
	newSeg, err := c.newGrowingSegment()
	if err != nil {
		c.mu.Unlock()
		return fmt.Errorf("collection flush: %w", err)
	}
	c.growing = newSeg
	c.mu.Unlock()

	done := make(chan error, 1)
	c.sealCh <- sealRequest{segment: old, done: done}
	return <-done
}

// Close stops the background sealer and releases resources, including any
// memory budget held by sealed segments.
func (c *Collection) Close() error {
	c.cancelF()
	c.wg.Wait()

	c.mu.Lock()
	if c.budget != nil {
		for _, n := range c.pinnedSegSize {
			c.budget.Release(n)
		}
	}
	c.pinnedSegSize = nil
	c.pinnedBytes = 0
	c.mu.Unlock()
	return nil
}

// sealerLoop processes seal requests sequentially.
func (c *Collection) sealerLoop(ctx context.Context) {
	defer c.wg.Done()
	for {
		select {
		case <-ctx.Done():
			return
		case req := <-c.sealCh:
			err := c.sealSegment(req.segment)
			req.done <- err
		}
	}
}

// sealSegment converts a growing segment to a sealed segment.
func (c *Collection) sealSegment(g *GrowingSegment) error {
	start := time.Now()
	entries := g.Entries()
	if len(entries) == 0 {
		return nil
	}

	// Reserve memory budget for the soon-to-be-sealed segment before doing
	// the expensive quantization work. If the budget is exhausted the seal
	// fails fast rather than producing data we cannot host.
	estBytes := memory.EstimateSegmentBytes(len(entries), c.dim, c.bitWidth)
	if c.budget != nil && estBytes > 0 {
		if err := c.budget.Acquire(context.Background(), estBytes); err != nil {
			return fmt.Errorf("seal segment %s: budget acquire: %w", g.ID(), err)
		}
	}

	sealed, err := Seal(g.ID(), entries, SealedSegmentConfig{
		ID:       g.ID(),
		Dim:      c.dim,
		BitWidth: c.bitWidth,
		Rotator:  c.rotator,
		Codebook: c.cb,
	})
	if err != nil {
		if c.budget != nil && estBytes > 0 {
			c.budget.Release(estBytes)
		}
		return fmt.Errorf("seal segment %s: %w", g.ID(), err)
	}

	c.mu.Lock()
	c.sealed = append(c.sealed, sealed)
	if estBytes > 0 {
		c.pinnedSegSize[sealed.ID()] = estBytes
		c.pinnedBytes += estBytes
	}
	// The sealing segment's copies are newer than any sealed copy of the
	// same id, so lift the masks that protected reads while these ids
	// lived in the growing segment. A stale copy in an older sealed
	// segment becomes reachable again, which search tolerates via
	// per-id dedup; compaction will remove it permanently.
	for _, entry := range entries {
		c.tombstones.Remove(entry.ID)
	}
	c.mu.Unlock()

	c.logger.Info("segment sealed",
		"segment", g.ID(),
		"vectors", sealed.Count(),
		"bytes_pinned", estBytes,
		"duration", time.Since(start),
	)
	if c.onSealed != nil {
		c.onSealed(sealed.ID(), sealed.Count(), estBytes)
	}
	return nil
}

// mergeTopK merges multiple result lists and returns the top-K by score (descending).
func mergeTopK(results []SearchResult, topK int) []SearchResult {
	if len(results) <= topK {
		sortResultsDesc(results)
		return results
	}

	// Use a min-heap to find top-K.
	h := make(minHeap, 0, topK)
	for _, r := range results {
		if len(h) < topK {
			h = append(h, r)
			if len(h) == topK {
				buildMinHeap(&h)
			}
		} else if r.Score > h[0].Score {
			h[0] = r
			heapifyDown(&h, 0)
		}
	}

	sortResultsDesc(h)
	return h
}

// sortResultsDesc sorts results by descending score using insertion sort (good for small K).
func sortResultsDesc(results []SearchResult) {
	for i := 1; i < len(results); i++ {
		key := results[i]
		j := i - 1
		for j >= 0 && results[j].Score < key.Score {
			results[j+1] = results[j]
			j--
		}
		results[j+1] = key
	}
}

func buildMinHeap(h *minHeap) {
	n := len(*h)
	for i := n/2 - 1; i >= 0; i-- {
		heapifyDown(h, i)
	}
}

func heapifyDown(h *minHeap, i int) {
	n := len(*h)
	for {
		smallest := i
		l, r := 2*i+1, 2*i+2
		if l < n && (*h)[l].Score < (*h)[smallest].Score {
			smallest = l
		}
		if r < n && (*h)[r].Score < (*h)[smallest].Score {
			smallest = r
		}
		if smallest == i {
			break
		}
		(*h)[i], (*h)[smallest] = (*h)[smallest], (*h)[i]
		i = smallest
	}
}

// dedupByID keeps the best-scoring result per id. An upserted id can
// transiently have copies in more than one segment until compaction.
func dedupByID(results []SearchResult) []SearchResult {
	best := make(map[string]int, len(results))
	out := results[:0]
	for _, r := range results {
		if i, ok := best[r.ID]; ok {
			if r.Score > out[i].Score {
				out[i] = r
			}
			continue
		}
		best[r.ID] = len(out)
		out = append(out, r)
	}
	return out
}
