package memory

// EstimateSegmentBytes returns an approximate resident byte cost for a sealed
// segment containing `count` vectors of dimension `dim` quantized at
// `bitWidth`. The estimate covers:
//
//   - Packed quantizer codes: ceil(dim*bitWidth/8) bytes per vector.
//   - Per-vector norm (float32): 4 bytes per vector.
//   - Per-vector ID: averageIDBytes per vector (rough heuristic — IDs are
//     short ULIDs/UUIDs in practice).
//
// It deliberately excludes amortized rotator/codebook bookkeeping because
// those are shared by every segment in a collection and are negligible
// compared to the per-vector arrays once `count` is large.
func EstimateSegmentBytes(count, dim, bitWidth int) int64 {
	if count <= 0 || dim <= 0 || bitWidth <= 0 {
		return 0
	}
	codeBytes := int64((dim*bitWidth + 7) / 8)
	const normBytes = 4
	const averageIDBytes = 24
	return int64(count) * (codeBytes + normBytes + averageIDBytes)
}
