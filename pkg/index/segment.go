// Package index implements the segment-based vector index architecture
// for the TurboDB engine. It provides growing (mutable) and sealed
// (immutable, quantized) segments managed by a Collection.
package index

import (
	"fmt"
	"time"
)

// SegmentType distinguishes growing from sealed segments.
type SegmentType int

const (
	// SegmentTypeGrowing is a mutable, append-only segment storing raw vectors.
	SegmentTypeGrowing SegmentType = iota
	// SegmentTypeSealed is an immutable segment storing quantized codes.
	SegmentTypeSealed
)

func (t SegmentType) String() string {
	switch t {
	case SegmentTypeGrowing:
		return "growing"
	case SegmentTypeSealed:
		return "sealed"
	default:
		return fmt.Sprintf("unknown(%d)", int(t))
	}
}

// VectorEntry represents a single vector stored in a growing segment.
type VectorEntry struct {
	ID       string
	Values   []float32
	Metadata map[string]string
}

// SearchResult represents a single result from a segment search.
type SearchResult struct {
	ID       string
	Score    float32
	Metadata map[string]string
}

// SegmentInfo holds metadata about a segment.
type SegmentInfo struct {
	ID        string
	Type      SegmentType
	Count     int
	Dim       int
	BitWidth  int
	CreatedAt time.Time
	SealedAt  time.Time // Zero for growing segments.
	SizeBytes int64
}

// Segment is the interface shared by growing and sealed segments.
type Segment interface {
	// ID returns the unique segment identifier.
	ID() string
	// Type returns whether this is a growing or sealed segment.
	Type() SegmentType
	// Count returns the number of vectors in this segment.
	Count() int
	// Info returns segment metadata.
	Info() SegmentInfo
	// Search returns the top-K results for a query vector, excluding tombstoned IDs.
	Search(query []float32, topK int, tombstones *TombstoneLog) ([]SearchResult, error)
	// Contains reports whether a vector ID exists in this segment.
	Contains(id string) bool
}
