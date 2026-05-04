package memory

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"
)

func TestBudgetUnlimited(t *testing.T) {
	t.Parallel()
	b := NewBudget(0)
	if !b.Unlimited() {
		t.Fatalf("expected unlimited budget")
	}
	if got := b.Capacity(); got != 0 {
		t.Errorf("Capacity() = %d, want 0 (sentinel)", got)
	}
	if err := b.Acquire(context.Background(), 1<<40); err != nil {
		t.Errorf("Acquire on unlimited budget: %v", err)
	}
	if got := b.Used(); got != 1<<40 {
		t.Errorf("Used() = %d, want %d", got, int64(1<<40))
	}
	b.Release(1 << 40)
	if got := b.Used(); got != 0 {
		t.Errorf("Used() after release = %d, want 0", got)
	}
}

func TestBudgetAcquireRelease(t *testing.T) {
	t.Parallel()
	b := NewBudget(1024)
	if err := b.Acquire(context.Background(), 256); err != nil {
		t.Fatalf("Acquire 256: %v", err)
	}
	if got := b.Used(); got != 256 {
		t.Errorf("Used = %d, want 256", got)
	}
	if err := b.Acquire(context.Background(), 768); err != nil {
		t.Fatalf("Acquire 768: %v", err)
	}
	if got := b.Used(); got != 1024 {
		t.Errorf("Used = %d, want 1024", got)
	}
	b.Release(256)
	if got := b.Used(); got != 768 {
		t.Errorf("Used after release = %d, want 768", got)
	}
}

func TestBudgetTryAcquire(t *testing.T) {
	t.Parallel()
	b := NewBudget(100)
	if !b.TryAcquire(60) {
		t.Fatalf("TryAcquire(60) = false, want true")
	}
	if b.TryAcquire(50) {
		t.Errorf("TryAcquire(50) succeeded, but only 40 bytes free")
	}
	if !b.TryAcquire(40) {
		t.Errorf("TryAcquire(40) = false, want true")
	}
	if got := b.Used(); got != 100 {
		t.Errorf("Used = %d, want 100", got)
	}
}

func TestBudgetAcquireBlocksUntilRelease(t *testing.T) {
	t.Parallel()
	b := NewBudget(100)
	if err := b.Acquire(context.Background(), 100); err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	wg.Add(1)
	acquired := make(chan struct{})
	go func() {
		defer wg.Done()
		if err := b.Acquire(context.Background(), 50); err != nil {
			t.Errorf("Acquire blocked: %v", err)
			return
		}
		close(acquired)
	}()

	select {
	case <-acquired:
		t.Fatalf("Acquire returned before Release")
	case <-time.After(20 * time.Millisecond):
	}

	b.Release(50)
	select {
	case <-acquired:
	case <-time.After(time.Second):
		t.Fatalf("Acquire did not unblock after Release")
	}
	wg.Wait()
}

func TestBudgetAcquireCancellation(t *testing.T) {
	t.Parallel()
	b := NewBudget(10)
	if err := b.Acquire(context.Background(), 10); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()

	err := b.Acquire(ctx, 5)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected DeadlineExceeded, got %v", err)
	}
	// Used must not have been bumped on a failed acquire.
	if got := b.Used(); got != 10 {
		t.Errorf("Used = %d, want 10 (no leak on failed acquire)", got)
	}
}

func TestBudgetRejectInvalidSize(t *testing.T) {
	t.Parallel()
	b := NewBudget(100)
	if err := b.Acquire(context.Background(), 0); err == nil {
		t.Errorf("expected error for n=0")
	}
	if err := b.Acquire(context.Background(), -1); err == nil {
		t.Errorf("expected error for n<0")
	}
	if err := b.Acquire(context.Background(), 200); err == nil {
		t.Errorf("expected error for n>capacity")
	}
}

func TestBudgetReleaseOverflowGuard(t *testing.T) {
	t.Parallel()
	b := NewBudget(100)
	if err := b.Acquire(context.Background(), 50); err != nil {
		t.Fatal(err)
	}
	// Releasing more than acquired should clamp at zero, not underflow.
	b.Release(100)
	if got := b.Used(); got != 0 {
		t.Errorf("Used after over-release = %d, want 0", got)
	}
}
