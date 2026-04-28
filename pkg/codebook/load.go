package codebook

import (
	"context"
	"encoding/json"
	"fmt"
)

// precomputedEntry is the JSON structure for embedded codebook data.
type precomputedEntry struct {
	Dim       int       `json:"dim"`
	BitWidth  int       `json:"bit_width"`
	Centroids []float64 `json:"centroids"`
}

// defaultCache backs the package-level Load / ClearCache helpers.
// Tests or multi-tenant servers that want isolation should construct their
// own Cache via NewCache and call Cache.Get directly.
var defaultCache = NewCache()

func cacheKey(dim, bitWidth int) string {
	return fmt.Sprintf("d%d_b%d", dim, bitWidth)
}

// Load returns a codebook for the given dimensionality and bit-width using
// the package-level default cache. Convenience wrapper around
// defaultCache.Get with context.Background.
func Load(dim, bitWidth int) (*Codebook, error) {
	return defaultCache.Get(context.Background(), dim, bitWidth)
}

// LoadCtx is like Load but accepts a context, used for any on-the-fly
// Lloyd-Max generation triggered by a cache miss.
func LoadCtx(ctx context.Context, dim, bitWidth int) (*Codebook, error) {
	return defaultCache.Get(ctx, dim, bitWidth)
}

// Generate creates a new codebook using Lloyd-Max for the given dim and bitWidth.
// The context allows callers to cancel or time out the generation.
func Generate(ctx context.Context, dim, bitWidth int) (*Codebook, error) {
	density, err := DensityForDim(dim)
	if err != nil {
		return nil, fmt.Errorf("codebook: density for d=%d: %w", dim, err)
	}
	cfg := DefaultLloydMaxConfig(density, bitWidth)
	result, err := SolveLloydMax(ctx, cfg)
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

// ClearCache evicts all entries from the package-level default cache.
// Useful in tests; for isolated caches, use Cache.Clear directly.
func ClearCache() {
	defaultCache.Clear()
}
