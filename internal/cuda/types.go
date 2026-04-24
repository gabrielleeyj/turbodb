package cuda

// SearchResult holds a single search result (vector ID + similarity score).
type SearchResult struct {
	ID    int64
	Score float32
}

// DeviceInfo holds GPU device information.
type DeviceInfo struct {
	FreeBytes  uint64
	TotalBytes uint64
	ComputeCapability int // major*10 + minor, e.g. 80 for A100
}
