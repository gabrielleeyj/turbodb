package pgproto

// This file defines the typed payloads for each opcode and their encode/decode
// functions. Postgres item pointers (ctid) are represented as a uint64 TID.

// BuildBegin starts an index build for a collection.
type BuildBegin struct {
	Collection  string
	Dim         uint32
	BitWidth    uint8
	Variant     uint8 // 0 = MSE
	RotatorSeed uint64
}

// Encode serializes the message body (without frame/version headers).
func (m BuildBegin) Encode() []byte {
	w := writer{}
	w.str(m.Collection)
	w.u32(m.Dim)
	w.u8(m.BitWidth)
	w.u8(m.Variant)
	w.u64(m.RotatorSeed)
	return w.buf
}

// DecodeBuildBegin parses a BuildBegin payload.
func DecodeBuildBegin(p []byte) (BuildBegin, error) {
	r := reader{buf: p}
	m := BuildBegin{
		Collection:  r.str(),
		Dim:         r.u32(),
		BitWidth:    r.u8(),
		Variant:     r.u8(),
		RotatorSeed: r.u64(),
	}
	return m, r.err
}

// VectorMsg carries a tid + vector, used for BUILD_VECTOR and INSERT.
type VectorMsg struct {
	TID    uint64
	Values []float32
}

// Encode serializes the message body (without frame/version headers).
func (m VectorMsg) Encode() []byte {
	w := writer{}
	w.u64(m.TID)
	w.vec(m.Values)
	return w.buf
}

// DecodeVectorMsg parses a VectorMsg payload.
func DecodeVectorMsg(p []byte) (VectorMsg, error) {
	r := reader{buf: p}
	m := VectorMsg{TID: r.u64(), Values: r.vec()}
	return m, r.err
}

// DeleteMsg tombstones a tid.
type DeleteMsg struct {
	TID uint64
}

// Encode serializes the message body (without frame/version headers).
func (m DeleteMsg) Encode() []byte {
	w := writer{}
	w.u64(m.TID)
	return w.buf
}

// DecodeDeleteMsg parses a DeleteMsg payload.
func DecodeDeleteMsg(p []byte) (DeleteMsg, error) {
	r := reader{buf: p}
	m := DeleteMsg{TID: r.u64()}
	return m, r.err
}

// SearchBegin starts a query-time scan.
type SearchBegin struct {
	Collection       string
	Query            []float32
	TopK             uint32
	OversearchFactor float32
	Rerank           bool
	Exact            bool
}

// Encode serializes the message body (without frame/version headers).
func (m SearchBegin) Encode() []byte {
	w := writer{}
	w.str(m.Collection)
	w.vec(m.Query)
	w.u32(m.TopK)
	w.f32(m.OversearchFactor)
	w.u8(boolToU8(m.Rerank))
	w.u8(boolToU8(m.Exact))
	return w.buf
}

// DecodeSearchBegin parses a SearchBegin payload.
func DecodeSearchBegin(p []byte) (SearchBegin, error) {
	r := reader{buf: p}
	m := SearchBegin{
		Collection:       r.str(),
		Query:            r.vec(),
		TopK:             r.u32(),
		OversearchFactor: r.f32(),
		Rerank:           r.u8() != 0,
		Exact:            r.u8() != 0,
	}
	return m, r.err
}

// ResultMsg is one search result row (reply to SEARCH_NEXT). Done signals the
// end of the result stream.
type ResultMsg struct {
	TID   uint64
	Score float32
	Done  bool
}

// Encode serializes the message body (without frame/version headers).
func (m ResultMsg) Encode() []byte {
	w := writer{}
	w.u64(m.TID)
	w.f32(m.Score)
	w.u8(boolToU8(m.Done))
	return w.buf
}

// DecodeResultMsg parses a ResultMsg payload.
func DecodeResultMsg(p []byte) (ResultMsg, error) {
	r := reader{buf: p}
	m := ResultMsg{TID: r.u64(), Score: r.f32(), Done: r.u8() != 0}
	return m, r.err
}

// StatsReply reports collection statistics (reply to STATS).
type StatsReply struct {
	VectorCount    uint64
	SealedSegments uint32
	GrowingSegment uint32
	PinnedBytes    uint64
}

// Encode serializes the message body (without frame/version headers).
func (m StatsReply) Encode() []byte {
	w := writer{}
	w.u64(m.VectorCount)
	w.u32(m.SealedSegments)
	w.u32(m.GrowingSegment)
	w.u64(m.PinnedBytes)
	return w.buf
}

// DecodeStatsReply parses a StatsReply payload.
func DecodeStatsReply(p []byte) (StatsReply, error) {
	r := reader{buf: p}
	m := StatsReply{
		VectorCount:    r.u64(),
		SealedSegments: r.u32(),
		GrowingSegment: r.u32(),
		PinnedBytes:    r.u64(),
	}
	return m, r.err
}

// EncodeError builds an OpError payload from a message string.
func EncodeError(msg string) []byte {
	w := writer{}
	w.str(msg)
	return w.buf
}

// DecodeError parses an OpError payload.
func DecodeError(p []byte) string {
	r := reader{buf: p}
	return r.str()
}

func boolToU8(b bool) uint8 {
	if b {
		return 1
	}
	return 0
}
