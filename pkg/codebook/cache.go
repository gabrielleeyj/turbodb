package codebook

import (
	"context"
	"fmt"
	"sync"
)

// Cache is an injectable codebook cache. Multiple goroutines may share a Cache
// safely; lookups use double-checked locking so concurrent Get calls for the
// same key collapse to a single load.
//
// A nil-receiver Cache is not valid; use NewCache.
type Cache struct {
	mu sync.RWMutex
	m  map[string]*Codebook
}

// NewCache returns an empty codebook cache.
func NewCache() *Cache {
	return &Cache{m: make(map[string]*Codebook)}
}

// Get returns the codebook for the given (dim, bitWidth), loading and caching
// it on first request. Behaviour mirrors the package-level Load: precomputed
// embedded assets are tried first, then Lloyd-Max generation as a fallback.
func (c *Cache) Get(ctx context.Context, dim, bitWidth int) (*Codebook, error) {
	key := cacheKey(dim, bitWidth)

	c.mu.RLock()
	if cb, ok := c.m[key]; ok {
		c.mu.RUnlock()
		return cb, nil
	}
	c.mu.RUnlock()

	c.mu.Lock()
	if cb, ok := c.m[key]; ok {
		c.mu.Unlock()
		return cb, nil
	}
	c.mu.Unlock()

	cb, err := loadPrecomputed(dim, bitWidth)
	if err != nil {
		cb, err = Generate(ctx, dim, bitWidth)
		if err != nil {
			return nil, fmt.Errorf("codebook: failed to generate d=%d b=%d: %w", dim, bitWidth, err)
		}
	}

	c.mu.Lock()
	if existing, ok := c.m[key]; ok {
		c.mu.Unlock()
		return existing, nil
	}
	c.m[key] = cb
	c.mu.Unlock()
	return cb, nil
}

// Clear evicts all cached codebooks. Useful in tests.
func (c *Cache) Clear() {
	c.mu.Lock()
	c.m = make(map[string]*Codebook)
	c.mu.Unlock()
}

// Len returns the number of cached entries.
func (c *Cache) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.m)
}
