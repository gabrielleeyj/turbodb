// Package memory implements byte-level resource budgeting for the TurboDB
// engine. Its primary export is Budget, a weighted semaphore over a fixed pool
// of bytes. Sealed segments and other long-lived allocations acquire from the
// budget at construction and release on close, giving the engine a single
// place to bound resident memory and (in future revisions) decide when to
// spill to host pinned memory or NVMe.
//
// The CPU MVP keeps all sealed segments resident. The Budget is therefore
// best understood as an accounting + admission-control layer: callers that
// would push the system over its configured ceiling block (or fail) instead
// of silently OOMing.
package memory

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"

	"golang.org/x/sync/semaphore"
)

// Budget bounds the total bytes pinned across a set of allocators. A capacity
// of zero (the default) creates an "unlimited" budget that still tracks
// usage but never blocks — useful for tests and for environments where the
// operator wants observability without admission control.
type Budget struct {
	capacity int64

	// sem is nil for unlimited budgets.
	sem *semaphore.Weighted

	// used is maintained alongside sem to provide cheap Used() reads and to
	// tolerate Release calls that exceed the actual outstanding amount
	// (clamping at zero rather than underflowing).
	used atomic.Int64

	// releaseMu serializes Release so the clamp-at-zero guard observes a
	// consistent (Used, sem) pair.
	releaseMu sync.Mutex
}

// NewBudget returns a Budget with the given byte capacity. capacityBytes <= 0
// means unlimited.
func NewBudget(capacityBytes int64) *Budget {
	if capacityBytes <= 0 {
		return &Budget{capacity: 0}
	}
	return &Budget{
		capacity: capacityBytes,
		sem:      semaphore.NewWeighted(capacityBytes),
	}
}

// Capacity returns the configured byte capacity. Zero indicates an unlimited
// budget.
func (b *Budget) Capacity() int64 { return b.capacity }

// Unlimited reports whether the budget enforces an upper bound.
func (b *Budget) Unlimited() bool { return b.sem == nil }

// Used returns the current bytes-in-use figure.
func (b *Budget) Used() int64 { return b.used.Load() }

// Acquire reserves n bytes, blocking until the request can be satisfied or
// ctx is canceled. n must be in (0, Capacity] for bounded budgets, or > 0
// for unlimited budgets.
func (b *Budget) Acquire(ctx context.Context, n int64) error {
	if n <= 0 {
		return fmt.Errorf("memory: acquire size must be > 0, got %d", n)
	}
	if !b.Unlimited() && n > b.capacity {
		return fmt.Errorf("memory: acquire size %d exceeds capacity %d", n, b.capacity)
	}
	if b.sem != nil {
		if err := b.sem.Acquire(ctx, n); err != nil {
			return err
		}
	}
	b.used.Add(n)
	return nil
}

// TryAcquire is a non-blocking variant that returns false if n bytes are not
// immediately available.
func (b *Budget) TryAcquire(n int64) bool {
	if n <= 0 {
		return false
	}
	if !b.Unlimited() && n > b.capacity {
		return false
	}
	if b.sem != nil && !b.sem.TryAcquire(n) {
		return false
	}
	b.used.Add(n)
	return true
}

// Release returns n bytes to the budget. Releasing more than is currently
// outstanding clamps at zero rather than underflowing — guarding against
// double-frees during shutdown paths.
func (b *Budget) Release(n int64) {
	if n <= 0 {
		return
	}
	b.releaseMu.Lock()
	defer b.releaseMu.Unlock()

	have := b.used.Load()
	if n > have {
		n = have
	}
	if n == 0 {
		return
	}
	b.used.Add(-n)
	if b.sem != nil {
		b.sem.Release(n)
	}
}
