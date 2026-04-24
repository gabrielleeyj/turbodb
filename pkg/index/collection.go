package index

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/gabrielleeyj/turbodb/pkg/codebook"
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

	// Sealing coordination.
	sealCh   chan sealRequest
	pendingWg sync.WaitGroup // Tracks in-flight seal operations.
	cancelF  context.CancelFunc
	wg       sync.WaitGroup
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

	// Check if we need to seal.
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
	found := c.growing.Contains(id)
	if !found {
		for _, s := range c.sealed {
			if s.Contains(id) {
				found = true
				break
			}
		}
	}
	c.mu.RUnlock()

	if !found {
		return fmt.Errorf("collection delete: vector ID %q not found", id)
	}

	c.tombstones.Delete(id)
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

	// Search growing segment.
	go func() {
		results, err := growing.Search(query, topK, c.tombstones)
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

	return mergeTopK(allResults, topK), nil
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
	c.mu.RUnlock()

	return CollectionStats{
		VectorCount:         totalVectors,
		GrowingSegmentCount: 1,
		SealedSegmentCount:  sealedCount,
		TombstoneCount:      c.tombstones.Count(),
	}
}

// CollectionStats holds statistics about a collection.
type CollectionStats struct {
	VectorCount         int
	GrowingSegmentCount int
	SealedSegmentCount  int
	TombstoneCount      int
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

// Close stops the background sealer and releases resources.
func (c *Collection) Close() error {
	c.cancelF()
	c.wg.Wait()
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

	sealed, err := Seal(g.ID(), entries, SealedSegmentConfig{
		ID:       g.ID(),
		Dim:      c.dim,
		BitWidth: c.bitWidth,
		Rotator:  c.rotator,
		Codebook: c.cb,
	})
	if err != nil {
		return fmt.Errorf("seal segment %s: %w", g.ID(), err)
	}

	c.mu.Lock()
	c.sealed = append(c.sealed, sealed)
	c.mu.Unlock()

	c.logger.Info("segment sealed",
		"segment", g.ID(),
		"vectors", sealed.Count(),
		"duration", time.Since(start),
	)
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
