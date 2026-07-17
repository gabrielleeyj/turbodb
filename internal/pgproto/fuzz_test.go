package pgproto

import (
	"bytes"
	"testing"
)

// FuzzReadFrame exercises the frame reader and every payload decoder with
// arbitrary bytes; none may panic, and successfully framed payloads must be
// safe to hand to all decoders.
func FuzzReadFrame(f *testing.F) {
	// Seeds: valid frames for a spread of opcodes plus adversarial prefixes.
	seed := func(op Opcode, payload []byte) []byte {
		var buf bytes.Buffer
		_ = WriteFrame(&buf, op, payload)
		return buf.Bytes()
	}
	f.Add(seed(OpBuildBegin, BuildBegin{Collection: "docs", Dim: 8, BitWidth: 4}.Encode()))
	f.Add(seed(OpInsert, VectorMsg{TID: 1, Values: []float32{1, 2}}.Encode()))
	f.Add(seed(OpDelete, DeleteMsg{TID: 1}.Encode()))
	f.Add(seed(OpSearchBegin, SearchBegin{Collection: "c", Query: []float32{1}, TopK: 1}.Encode()))
	f.Add(seed(OpResult, ResultMsg{TID: 1, Score: 0.5}.Encode()))
	f.Add(seed(OpStats, StatsReply{VectorCount: 1}.Encode()))
	f.Add([]byte{0xFF, 0xFF, 0xFF, 0xFF})       // huge length prefix
	f.Add([]byte{0, 0, 0, 2, 0, 1})             // body shorter than minimum
	f.Add([]byte{0, 0, 0, 4, 0, 1, 0x99, 0x00}) // bad schema version

	f.Fuzz(func(t *testing.T, data []byte) {
		frame, err := ReadFrame(bytes.NewReader(data))
		if err != nil {
			return
		}
		// Any successfully framed payload must be safe for every decoder.
		_, _ = DecodeBuildBegin(frame.Payload)
		_, _ = DecodeVectorMsg(frame.Payload)
		_, _ = DecodeDeleteMsg(frame.Payload)
		_, _ = DecodeSearchBegin(frame.Payload)
		_, _ = DecodeResultMsg(frame.Payload)
		_, _ = DecodeStatsReply(frame.Payload)
		_ = DecodeError(frame.Payload)
	})
}
