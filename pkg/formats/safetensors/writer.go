package safetensors

import (
	"encoding/binary"
	"fmt"
	"io"
)

// Writer streams a SafeTensors file to an io.Writer without buffering tensor
// data in memory. Usage is two-phase: declare every tensor with Declare, call
// Commit to flush the header, then stream each tensor's bytes with Write in the
// same order they were declared.
type Writer struct {
	w         io.Writer
	metadata  map[string]string
	order     []string
	tensors   map[string]TensorInfo
	cursor    int64 // next data-section offset to assign
	committed bool
	writeIdx  int   // index into order of the next expected Write
	written   int64 // bytes written into the data section so far
}

// NewWriter creates a Writer targeting w. metadata may be nil.
func NewWriter(w io.Writer, metadata map[string]string) *Writer {
	md := make(map[string]string, len(metadata))
	for k, v := range metadata {
		md[k] = v
	}
	return &Writer{
		w:        w,
		metadata: md,
		tensors:  make(map[string]TensorInfo),
	}
}

// Declare registers a tensor and reserves its contiguous byte range. Tensors
// occupy the data section in declaration order. It must be called before
// Commit.
func (w *Writer) Declare(name string, dtype Dtype, shape []int64) error {
	if w.committed {
		return fmt.Errorf("safetensors: cannot declare %q after Commit", name)
	}
	if _, dup := w.tensors[name]; dup {
		return fmt.Errorf("safetensors: tensor %q already declared", name)
	}
	elemSize, err := dtype.ByteSize()
	if err != nil {
		return err
	}
	count, err := numElements(shape)
	if err != nil {
		return err
	}
	nbytes := count * int64(elemSize)
	info := TensorInfo{
		Dtype:       dtype,
		Shape:       append([]int64(nil), shape...),
		DataOffsets: [2]int64{w.cursor, w.cursor + nbytes},
	}
	w.tensors[name] = info
	w.order = append(w.order, name)
	w.cursor += nbytes
	return nil
}

// Commit writes the length-prefixed JSON header. After Commit, only Write and
// Close may be called.
func (w *Writer) Commit() error {
	if w.committed {
		return fmt.Errorf("safetensors: already committed")
	}
	headerJSON, err := buildHeaderJSON(w.tensors, w.metadata)
	if err != nil {
		return err
	}
	var lenBuf [headerLenSize]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(headerJSON)))
	if _, err := w.w.Write(lenBuf[:]); err != nil {
		return fmt.Errorf("safetensors: write header length: %w", err)
	}
	if _, err := w.w.Write(headerJSON); err != nil {
		return fmt.Errorf("safetensors: write header: %w", err)
	}
	w.committed = true
	return nil
}

// Write streams one tensor's raw little-endian bytes. Tensors must be written
// in declaration order and match the declared byte length exactly.
func (w *Writer) Write(name string, data []byte) error {
	if !w.committed {
		return fmt.Errorf("safetensors: Commit must be called before Write")
	}
	if w.writeIdx >= len(w.order) {
		return fmt.Errorf("safetensors: unexpected Write(%q): all tensors already written", name)
	}
	expected := w.order[w.writeIdx]
	if name != expected {
		return fmt.Errorf("safetensors: out-of-order Write: expected %q, got %q", expected, name)
	}
	if want := w.tensors[name].nbytes(); int64(len(data)) != want {
		return fmt.Errorf("safetensors: tensor %q expects %d bytes, got %d", name, want, len(data))
	}
	if _, err := w.w.Write(data); err != nil {
		return fmt.Errorf("safetensors: write tensor %q: %w", name, err)
	}
	w.writeIdx++
	w.written += int64(len(data))
	return nil
}

// Close verifies all declared tensors were written. It does not close the
// underlying io.Writer.
func (w *Writer) Close() error {
	if !w.committed {
		return fmt.Errorf("safetensors: Close before Commit")
	}
	if w.writeIdx != len(w.order) {
		return fmt.Errorf("safetensors: %d of %d tensors written before Close", w.writeIdx, len(w.order))
	}
	return nil
}

// Save is a convenience that writes a complete SafeTensors file from in-memory
// tensors. Tensors are emitted in sorted name order. Prefer the streaming
// Declare/Commit/Write API for files larger than RAM.
func Save(w io.Writer, tensors []*Tensor, metadata map[string]string) error {
	sw := NewWriter(w, metadata)
	byName := make(map[string][]byte, len(tensors))
	for _, t := range tensors {
		if err := sw.Declare(t.Name, t.Info.Dtype, t.Info.Shape); err != nil {
			return err
		}
		byName[t.Name] = t.Data
	}
	if err := sw.Commit(); err != nil {
		return err
	}
	for _, name := range sw.order {
		if err := sw.Write(name, byName[name]); err != nil {
			return err
		}
	}
	return sw.Close()
}
