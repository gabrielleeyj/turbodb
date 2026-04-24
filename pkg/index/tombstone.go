package index

import "sync"

// TombstoneLog tracks deleted vector IDs. Deletions are soft — vectors
// are excluded from search results and removed permanently at compaction.
// TombstoneLog is safe for concurrent use.
type TombstoneLog struct {
	mu      sync.RWMutex
	deleted map[string]struct{}
}

// NewTombstoneLog creates an empty tombstone log.
func NewTombstoneLog() *TombstoneLog {
	return &TombstoneLog{deleted: make(map[string]struct{})}
}

// NewTombstoneLogFrom creates a tombstone log pre-populated with the given IDs.
func NewTombstoneLogFrom(ids []string) *TombstoneLog {
	t := &TombstoneLog{deleted: make(map[string]struct{}, len(ids))}
	for _, id := range ids {
		t.deleted[id] = struct{}{}
	}
	return t
}

// Delete marks a vector ID as deleted.
func (t *TombstoneLog) Delete(id string) {
	t.mu.Lock()
	t.deleted[id] = struct{}{}
	t.mu.Unlock()
}

// IsDeleted reports whether the given ID has been tombstoned.
func (t *TombstoneLog) IsDeleted(id string) bool {
	t.mu.RLock()
	_, ok := t.deleted[id]
	t.mu.RUnlock()
	return ok
}

// Count returns the number of tombstoned IDs.
func (t *TombstoneLog) Count() int {
	t.mu.RLock()
	n := len(t.deleted)
	t.mu.RUnlock()
	return n
}

// IDs returns a snapshot of all tombstoned vector IDs.
func (t *TombstoneLog) IDs() []string {
	t.mu.RLock()
	ids := make([]string, 0, len(t.deleted))
	for id := range t.deleted {
		ids = append(ids, id)
	}
	t.mu.RUnlock()
	return ids
}

// Remove un-tombstones an ID (used after compaction permanently removes the vector).
func (t *TombstoneLog) Remove(id string) {
	t.mu.Lock()
	delete(t.deleted, id)
	t.mu.Unlock()
}

// Clear removes all tombstones (used after a full compaction).
func (t *TombstoneLog) Clear() {
	t.mu.Lock()
	t.deleted = make(map[string]struct{})
	t.mu.Unlock()
}
