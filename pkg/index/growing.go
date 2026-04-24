package index

import (
	"container/heap"
	"fmt"
	"maps"
	"sync"
	"time"

	"github.com/gabrielleeyj/turbodb/pkg/codebook"
	"github.com/gabrielleeyj/turbodb/pkg/rotation"
)

// GrowingSegment is a mutable, append-only segment that stores raw vectors.
// Queries use brute-force inner-product search. It is safe for concurrent use.
type GrowingSegment struct {
	mu        sync.RWMutex
	id        string
	dim       int
	entries   []VectorEntry
	idIndex   map[string]int // id -> index in entries
	createdAt time.Time

	// Quantizer components stored for sealing.
	rotator  rotation.Rotator
	codebook *codebook.Codebook
	bitWidth int
}

// GrowingSegmentConfig holds configuration for creating a GrowingSegment.
type GrowingSegmentConfig struct {
	ID       string
	Dim      int
	Rotator  rotation.Rotator
	Codebook *codebook.Codebook
	BitWidth int
}

// NewGrowingSegment creates a new empty growing segment.
func NewGrowingSegment(cfg GrowingSegmentConfig) (*GrowingSegment, error) {
	if cfg.ID == "" {
		return nil, fmt.Errorf("growing segment: ID must not be empty")
	}
	if cfg.Dim < 1 {
		return nil, fmt.Errorf("growing segment: dim must be >= 1, got %d", cfg.Dim)
	}
	if cfg.Rotator == nil {
		return nil, fmt.Errorf("growing segment: rotator must not be nil")
	}
	if cfg.Codebook == nil {
		return nil, fmt.Errorf("growing segment: codebook must not be nil")
	}
	if cfg.BitWidth < 1 || cfg.BitWidth > 8 {
		return nil, fmt.Errorf("growing segment: bitWidth must be 1..8, got %d", cfg.BitWidth)
	}
	return &GrowingSegment{
		id:        cfg.ID,
		dim:       cfg.Dim,
		entries:   nil,
		idIndex:   make(map[string]int),
		createdAt: time.Now(),
		rotator:   cfg.Rotator,
		codebook:  cfg.Codebook,
		bitWidth:  cfg.BitWidth,
	}, nil
}

// ID returns the segment identifier.
func (g *GrowingSegment) ID() string { return g.id }

// Type returns SegmentTypeGrowing.
func (g *GrowingSegment) Type() SegmentType { return SegmentTypeGrowing }

// Count returns the number of vectors.
func (g *GrowingSegment) Count() int {
	g.mu.RLock()
	n := len(g.entries)
	g.mu.RUnlock()
	return n
}

// Info returns segment metadata.
func (g *GrowingSegment) Info() SegmentInfo {
	g.mu.RLock()
	n := len(g.entries)
	g.mu.RUnlock()
	return SegmentInfo{
		ID:        g.id,
		Type:      SegmentTypeGrowing,
		Count:     n,
		Dim:       g.dim,
		BitWidth:  g.bitWidth,
		CreatedAt: g.createdAt,
	}
}

// Insert adds a vector to the segment. Returns an error if the ID already exists
// or the dimension doesn't match.
func (g *GrowingSegment) Insert(entry VectorEntry) error {
	if len(entry.Values) != g.dim {
		return fmt.Errorf("growing segment: expected dim %d, got %d", g.dim, len(entry.Values))
	}
	if entry.ID == "" {
		return fmt.Errorf("growing segment: vector ID must not be empty")
	}

	// Copy values and metadata to ensure immutability of stored data.
	values := make([]float32, len(entry.Values))
	copy(values, entry.Values)

	var meta map[string]string
	if len(entry.Metadata) > 0 {
		meta = make(map[string]string, len(entry.Metadata))
		maps.Copy(meta, entry.Metadata)
	}

	stored := VectorEntry{ID: entry.ID, Values: values, Metadata: meta}

	g.mu.Lock()
	if _, exists := g.idIndex[entry.ID]; exists {
		g.mu.Unlock()
		return fmt.Errorf("growing segment: duplicate vector ID %q", entry.ID)
	}
	g.idIndex[entry.ID] = len(g.entries)
	g.entries = append(g.entries, stored)
	g.mu.Unlock()
	return nil
}

// Contains reports whether a vector ID exists in this segment.
func (g *GrowingSegment) Contains(id string) bool {
	g.mu.RLock()
	_, ok := g.idIndex[id]
	g.mu.RUnlock()
	return ok
}

// Search performs brute-force inner-product search, returning up to topK results.
// Tombstoned IDs are excluded. Results are sorted by descending score.
func (g *GrowingSegment) Search(query []float32, topK int, tombstones *TombstoneLog) ([]SearchResult, error) {
	if len(query) != g.dim {
		return nil, fmt.Errorf("growing segment search: expected dim %d, got %d", g.dim, len(query))
	}
	if topK < 1 {
		return nil, fmt.Errorf("growing segment search: topK must be >= 1")
	}

	g.mu.RLock()
	entries := g.entries
	g.mu.RUnlock()

	h := &minHeap{}
	heap.Init(h)

	for i := range entries {
		e := &entries[i]
		if tombstones != nil && tombstones.IsDeleted(e.ID) {
			continue
		}

		score := innerProduct(query, e.Values)

		if h.Len() < topK {
			heap.Push(h, SearchResult{ID: e.ID, Score: score, Metadata: e.Metadata})
		} else if score > (*h)[0].Score {
			(*h)[0] = SearchResult{ID: e.ID, Score: score, Metadata: e.Metadata}
			heap.Fix(h, 0)
		}
	}

	// Extract results in descending score order.
	results := make([]SearchResult, h.Len())
	for i := len(results) - 1; i >= 0; i-- {
		results[i] = heap.Pop(h).(SearchResult)
	}
	return results, nil
}

// Entries returns a snapshot of all vector entries (used during sealing).
func (g *GrowingSegment) Entries() []VectorEntry {
	g.mu.RLock()
	snapshot := make([]VectorEntry, len(g.entries))
	copy(snapshot, g.entries)
	g.mu.RUnlock()
	return snapshot
}

// Rotator returns the rotation matrix for this segment.
func (g *GrowingSegment) Rotator() rotation.Rotator { return g.rotator }

// CodebookRef returns the codebook for this segment.
func (g *GrowingSegment) CodebookRef() *codebook.Codebook { return g.codebook }

// BitWidthVal returns the quantization bit-width.
func (g *GrowingSegment) BitWidthVal() int { return g.bitWidth }

// innerProduct computes the dot product of two vectors.
func innerProduct(a, b []float32) float32 {
	var sum float64
	n := min(len(a), len(b))
	for i := range n {
		sum += float64(a[i]) * float64(b[i])
	}
	return float32(sum)
}

// minHeap is a min-heap of SearchResults ordered by Score.
// Used to efficiently maintain the top-K results.
type minHeap []SearchResult

func (h minHeap) Len() int            { return len(h) }
func (h minHeap) Less(i, j int) bool  { return h[i].Score < h[j].Score }
func (h minHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x any) { *h = append(*h, x.(SearchResult)) }
func (h *minHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}
