package pgproto

import (
	"encoding/binary"
	"fmt"
	"math"
)

// writer is a little-endian payload builder.
type writer struct {
	buf []byte
}

func (w *writer) u8(v uint8)    { w.buf = append(w.buf, v) }
func (w *writer) u32(v uint32)  { w.buf = binary.LittleEndian.AppendUint32(w.buf, v) }
func (w *writer) u64(v uint64)  { w.buf = binary.LittleEndian.AppendUint64(w.buf, v) }
func (w *writer) f32(v float32) { w.u32(math.Float32bits(v)) }

func (w *writer) str(s string) {
	w.u32(uint32(len(s)))
	w.buf = append(w.buf, s...)
}

func (w *writer) vec(v []float32) {
	w.u32(uint32(len(v)))
	for _, x := range v {
		w.f32(x)
	}
}

// reader is a little-endian payload parser that tracks the first error.
type reader struct {
	buf []byte
	off int
	err error
}

func (r *reader) need(n int) bool {
	if r.err != nil {
		return false
	}
	if r.off+n > len(r.buf) {
		r.err = fmt.Errorf("pgproto: truncated payload: need %d bytes at offset %d of %d", n, r.off, len(r.buf))
		return false
	}
	return true
}

func (r *reader) u8() uint8 {
	if !r.need(1) {
		return 0
	}
	v := r.buf[r.off]
	r.off++
	return v
}

func (r *reader) u32() uint32 {
	if !r.need(4) {
		return 0
	}
	v := binary.LittleEndian.Uint32(r.buf[r.off:])
	r.off += 4
	return v
}

func (r *reader) u64() uint64 {
	if !r.need(8) {
		return 0
	}
	v := binary.LittleEndian.Uint64(r.buf[r.off:])
	r.off += 8
	return v
}

func (r *reader) f32() float32 { return math.Float32frombits(r.u32()) }

func (r *reader) str() string {
	n := int(r.u32())
	if n < 0 || !r.need(n) {
		return ""
	}
	s := string(r.buf[r.off : r.off+n])
	r.off += n
	return s
}

func (r *reader) vec() []float32 {
	n := int(r.u32())
	if r.err != nil || n < 0 || !r.need(n*4) {
		return nil
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = r.f32()
	}
	return out
}
