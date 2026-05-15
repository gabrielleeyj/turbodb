package gguf

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"sort"
)

// Writer streams a GGUF file. Usage: set metadata and Declare every tensor,
// then Commit to flush the header and tensor-info table, then WriteData for
// each tensor in declaration order. Tensor data is aligned per the alignment
// option (default 32), matching ggml's expectations.
type Writer struct {
	w         io.Writer
	alignment int64
	metadata  map[string]Value
	metaOrder []string
	order     []string
	tensors   map[string]TensorInfo
	sizes     map[string]int64
	cursor    int64 // next data-section offset to assign
	committed bool
	writeIdx  int
	dataPos   int64 // bytes written into the data section so far
}

// NewWriter creates a GGUF Writer with the default alignment.
func NewWriter(w io.Writer) *Writer {
	return &Writer{
		w:         w,
		alignment: DefaultAlignment,
		metadata:  make(map[string]Value),
		tensors:   make(map[string]TensorInfo),
		sizes:     make(map[string]int64),
	}
}

// SetAlignment overrides the tensor-data alignment. Must be called before
// Commit. The value is also recorded as general.alignment metadata.
func (w *Writer) SetAlignment(a int64) {
	if a >= 1 {
		w.alignment = a
	}
}

// SetMetadata records a metadata key/value. Must be called before Commit.
func (w *Writer) SetMetadata(key string, v Value) {
	if _, exists := w.metadata[key]; !exists {
		w.metaOrder = append(w.metaOrder, key)
	}
	w.metadata[key] = v
}

// Declare registers a tensor with its element type, dimensions, and exact
// on-disk byte length, reserving an aligned offset in the data section.
func (w *Writer) Declare(name string, typ GGMLType, dims []uint64, nbytes int64) error {
	if w.committed {
		return fmt.Errorf("gguf: cannot declare %q after Commit", name)
	}
	if _, dup := w.tensors[name]; dup {
		return fmt.Errorf("gguf: tensor %q already declared", name)
	}
	if nbytes < 0 {
		return fmt.Errorf("gguf: tensor %q negative size %d", name, nbytes)
	}
	offset := align(w.cursor, w.alignment)
	w.tensors[name] = TensorInfo{
		Name:   name,
		Dims:   append([]uint64(nil), dims...),
		Type:   typ,
		Offset: uint64(offset),
	}
	w.sizes[name] = nbytes
	w.order = append(w.order, name)
	w.cursor = offset + nbytes
	return nil
}

// Commit writes the magic, header counts, metadata, tensor-info table, and the
// padding up to the first aligned data offset.
func (w *Writer) Commit() error {
	if w.committed {
		return fmt.Errorf("gguf: already committed")
	}
	// Ensure general.alignment metadata reflects the chosen alignment.
	w.SetMetadata(AlignmentKey, Uint32Value(uint32(w.alignment)))

	var buf bytes.Buffer
	putU32(&buf, Magic)
	putU32(&buf, Version)
	putU64(&buf, uint64(len(w.order)))
	putU64(&buf, uint64(len(w.metaOrder)))

	for _, key := range w.metaOrder {
		putString(&buf, key)
		if err := writeValue(&buf, w.metadata[key]); err != nil {
			return err
		}
	}
	for _, name := range w.order {
		ti := w.tensors[name]
		putString(&buf, name)
		putU32(&buf, uint32(len(ti.Dims)))
		for _, d := range ti.Dims {
			putU64(&buf, d)
		}
		putU32(&buf, uint32(ti.Type))
		putU64(&buf, ti.Offset)
	}

	headerLen := int64(buf.Len())
	dataBase := align(headerLen, w.alignment)
	pad := dataBase - headerLen
	buf.Write(make([]byte, pad))

	if _, err := w.w.Write(buf.Bytes()); err != nil {
		return fmt.Errorf("gguf: write header: %w", err)
	}
	w.committed = true
	return nil
}

// WriteData streams one tensor's raw bytes in declaration order, inserting
// alignment padding between tensors.
func (w *Writer) WriteData(name string, data []byte) error {
	if !w.committed {
		return fmt.Errorf("gguf: Commit must be called before WriteData")
	}
	if w.writeIdx >= len(w.order) {
		return fmt.Errorf("gguf: unexpected WriteData(%q): all tensors written", name)
	}
	expected := w.order[w.writeIdx]
	if name != expected {
		return fmt.Errorf("gguf: out-of-order WriteData: expected %q, got %q", expected, name)
	}
	if int64(len(data)) != w.sizes[name] {
		return fmt.Errorf("gguf: tensor %q expects %d bytes, got %d", name, w.sizes[name], len(data))
	}
	// Pad from current data position up to this tensor's aligned offset.
	target := int64(w.tensors[name].Offset)
	if pad := target - w.dataPos; pad > 0 {
		if _, err := w.w.Write(make([]byte, pad)); err != nil {
			return fmt.Errorf("gguf: write padding: %w", err)
		}
		w.dataPos += pad
	}
	if _, err := w.w.Write(data); err != nil {
		return fmt.Errorf("gguf: write tensor %q: %w", name, err)
	}
	w.dataPos += int64(len(data))
	w.writeIdx++
	return nil
}

// Close verifies all declared tensors were written.
func (w *Writer) Close() error {
	if !w.committed {
		return fmt.Errorf("gguf: Close before Commit")
	}
	if w.writeIdx != len(w.order) {
		return fmt.Errorf("gguf: %d of %d tensors written before Close", w.writeIdx, len(w.order))
	}
	return nil
}

func putU32(buf *bytes.Buffer, v uint32) {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	buf.Write(b[:])
}

func putU64(buf *bytes.Buffer, v uint64) {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], v)
	buf.Write(b[:])
}

func putString(buf *bytes.Buffer, s string) {
	putU64(buf, uint64(len(s)))
	buf.WriteString(s)
}

// writeValue serializes a metadata value (with its type tag).
func writeValue(buf *bytes.Buffer, v Value) error {
	putU32(buf, uint32(v.Type))
	return writeValueBody(buf, v)
}

func writeValueBody(buf *bytes.Buffer, v Value) error {
	switch v.Type {
	case mvUint8, mvInt8, mvBool:
		buf.WriteByte(byte(v.Num))
	case mvUint16, mvInt16:
		var b [2]byte
		binary.LittleEndian.PutUint16(b[:], uint16(v.Num))
		buf.Write(b[:])
	case mvUint32, mvInt32:
		putU32(buf, uint32(v.Num))
	case mvFloat32:
		putU32(buf, float32bits(float32(v.F64)))
	case mvUint64, mvInt64:
		putU64(buf, v.Num)
	case mvFloat64:
		putU64(buf, float64bits(v.F64))
	case mvString:
		putString(buf, v.Str)
	case mvArray:
		putU32(buf, uint32(v.ArrayType))
		putU64(buf, uint64(len(v.Array)))
		for _, e := range v.Array {
			if err := writeValueBody(buf, e); err != nil {
				return err
			}
		}
	default:
		return fmt.Errorf("gguf: cannot write metadata value type %d", v.Type)
	}
	return nil
}

// SortedMetadataKeys is a small helper for callers/tests that want a stable
// metadata ordering independent of insertion order.
func SortedMetadataKeys(keys []string) []string {
	out := append([]string(nil), keys...)
	sort.Strings(out)
	return out
}
