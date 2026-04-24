package cuda

import "sync"

// Pool manages a set of CUDA contexts for concurrent use.
// Each context owns a dedicated CUDA stream on the same device.
//
// Callers acquire a context with Get() and return it with Put().
// This avoids contention on a single CUDA stream.
type Pool struct {
	mu       sync.Mutex
	deviceID int
	contexts []Context
	maxSize  int
}

// NewPool creates a context pool for the given device.
// maxSize limits the number of concurrent contexts (and thus CUDA streams).
func NewPool(deviceID, maxSize int) *Pool {
	return &Pool{
		deviceID: deviceID,
		maxSize:  maxSize,
	}
}

// Get returns an available context from the pool, or creates a new one
// if the pool is empty and hasn't reached maxSize.
func (p *Pool) Get() (Context, error) {
	p.mu.Lock()
	if len(p.contexts) > 0 {
		ctx := p.contexts[len(p.contexts)-1]
		p.contexts = p.contexts[:len(p.contexts)-1]
		p.mu.Unlock()
		return ctx, nil
	}
	p.mu.Unlock()

	return NewContext(p.deviceID)
}

// Put returns a context to the pool for reuse.
// If the pool is at capacity, the context is closed instead.
func (p *Pool) Put(ctx Context) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if len(p.contexts) < p.maxSize {
		p.contexts = append(p.contexts, ctx)
	} else {
		ctx.Close()
	}
}

// Close destroys all contexts in the pool.
func (p *Pool) Close() {
	p.mu.Lock()
	defer p.mu.Unlock()

	for _, ctx := range p.contexts {
		ctx.Close()
	}
	p.contexts = nil
}

// Size returns the number of idle contexts in the pool.
func (p *Pool) Size() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.contexts)
}
