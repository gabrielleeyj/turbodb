package codebook

import (
	"encoding/json"
	"fmt"
	"sync"
)

// precomputedEntry is the JSON structure for embedded codebook data.
type precomputedEntry struct {
	Dim       int       `json:"dim"`
	BitWidth  int       `json:"bit_width"`
	Centroids []float64 `json:"centroids"`
}

var (
	cacheMu sync.RWMutex
	cache   = make(map[string]*Codebook)
)

func cacheKey(dim, bitWidth int) string {
	return fmt.Sprintf("d%d_b%d", dim, bitWidth)
}

// Load returns a codebook for the given dimensionality and bit-width.
// It checks the in-memory cache first, then precomputed embedded assets,
// and finally generates one on-the-fly using Lloyd-Max.
//
// Uses double-checked locking: the cache is re-checked after acquiring the
// write lock to prevent redundant loads when multiple goroutines race past
// the initial read-lock check.
func Load(dim, bitWidth int) (*Codebook, error) {
	key := cacheKey(dim, bitWidth)

	// Fast path: check cache under read lock.
	cacheMu.RLock()
	if cb, ok := cache[key]; ok {
		cacheMu.RUnlock()
		return cb, nil
	}
	cacheMu.RUnlock()

	// Slow path: acquire write lock and re-check before loading.
	cacheMu.Lock()
	if cb, ok := cache[key]; ok {
		cacheMu.Unlock()
		return cb, nil
	}
	cacheMu.Unlock()

	// Load outside the lock to avoid holding it during I/O.
	cb, err := loadPrecomputed(dim, bitWidth)
	if err != nil {
		cb, err = Generate(dim, bitWidth)
		if err != nil {
			return nil, fmt.Errorf("codebook: failed to generate d=%d b=%d: %w", dim, bitWidth, err)
		}
	}

	// Store under write lock; re-check in case another goroutine won the race.
	cacheMu.Lock()
	if existing, ok := cache[key]; ok {
		cacheMu.Unlock()
		return existing, nil
	}
	cache[key] = cb
	cacheMu.Unlock()
	return cb, nil
}

// Generate creates a new codebook using Lloyd-Max for the given dim and bitWidth.
func Generate(dim, bitWidth int) (*Codebook, error) {
	density := DensityForDim(dim)
	cfg := DefaultLloydMaxConfig(density, bitWidth)
	result, err := SolveLloydMax(cfg)
	if err != nil {
		return nil, err
	}
	return NewCodebook(dim, bitWidth, result.Centroids)
}

// loadPrecomputed attempts to load a codebook from embedded assets.
func loadPrecomputed(dim, bitWidth int) (*Codebook, error) {
	key := cacheKey(dim, bitWidth)
	filename := key + ".json"

	data, err := precomputedFS.ReadFile("precomputed/" + filename)
	if err != nil {
		return nil, fmt.Errorf("codebook: no precomputed data for %s: %w", key, err)
	}

	var entry precomputedEntry
	if err := json.Unmarshal(data, &entry); err != nil {
		return nil, fmt.Errorf("codebook: invalid precomputed data for %s: %w", key, err)
	}

	return NewCodebook(entry.Dim, entry.BitWidth, entry.Centroids)
}

// ClearCache removes all cached codebooks. Useful in tests.
func ClearCache() {
	cacheMu.Lock()
	cache = make(map[string]*Codebook)
	cacheMu.Unlock()
}
