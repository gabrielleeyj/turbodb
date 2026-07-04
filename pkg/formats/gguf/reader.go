package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"sort"
)

// maxStringLen bounds GGUF string lengths to guard against corrupt or hostile
// files declaring absurd allocations.
const maxStringLen = 1 << 30

// TensorInfo describes one tensor's name, shape, encoding, and byte offset
// within the data section.
type TensorInfo struct {
	Name   string
	Dims   []uint64
	Type   GGMLType
	Offset uint64 // relative to the start of the (aligned) data section
}

// Bounds for values read from untrusted files. They keep all downstream
// size arithmetic (element counts, byte lengths, file offsets) safely inside
// int64 range.
const (
	maxTensorElems  = 1 << 48
	maxTensorOffset = 1 << 62
	maxAlignment    = 1 << 20
)

// numElements returns the product of the dimensions. Bounded by the
// maxTensorElems check in readTensorInfo.
func (ti TensorInfo) numElements() uint64 {
	n := uint64(1)
	for _, d := range ti.Dims {
		n *= d
	}
	return n
}

// File provides read access to a parsed GGUF file.
type File struct {
	src       io.ReaderAt
	closer    io.Closer
	metadata  map[string]Value
	tensors   map[string]TensorInfo
	rawLen    map[string]int64 // on-disk byte length per tensor, from offset boundaries
	order     []string
	dataBase  int64
	alignment int64
}

// cursor is a sequential little-endian reader over an io.ReaderAt.
type cursor struct {
	src io.ReaderAt
	off int64
	err error
}

func (c *cursor) read(n int64) []byte {
	if c.err != nil {
		return nil
	}
	buf := make([]byte, n)
	if _, err := c.src.ReadAt(buf, c.off); err != nil {
		c.err = err
		return nil
	}
	c.off += n
	return buf
}

func (c *cursor) u32() uint32 {
	b := c.read(4)
	if b == nil {
		return 0
	}
	return binary.LittleEndian.Uint32(b)
}

func (c *cursor) u64() uint64 {
	b := c.read(8)
	if b == nil {
		return 0
	}
	return binary.LittleEndian.Uint64(b)
}

func (c *cursor) str() string {
	n := c.u64()
	if c.err != nil {
		return ""
	}
	if n > maxStringLen {
		c.err = fmt.Errorf("gguf: string length %d exceeds max %d", n, maxStringLen)
		return ""
	}
	b := c.read(int64(n))
	return string(b)
}

// Open memory-maps... actually opens the GGUF file at path and parses it.
func Open(path string) (*File, error) {
	fh, err := os.Open(path) // #nosec G304 -- caller-supplied path is this API's contract
	if err != nil {
		return nil, fmt.Errorf("gguf: open %q: %w", path, err)
	}
	info, err := fh.Stat()
	if err != nil {
		_ = fh.Close()
		return nil, fmt.Errorf("gguf: stat %q: %w", path, err)
	}
	f, err := NewReader(fh, info.Size())
	if err != nil {
		_ = fh.Close()
		return nil, err
	}
	f.closer = fh
	return f, nil
}

// NewReader parses GGUF structure from src (which must expose size bytes).
func NewReader(src io.ReaderAt, size int64) (*File, error) {
	c := &cursor{src: src}
	if magic := c.u32(); magic != Magic {
		return nil, fmt.Errorf("gguf: bad magic 0x%08x (want 0x%08x)", magic, Magic)
	}
	if version := c.u32(); version != Version {
		return nil, fmt.Errorf("gguf: unsupported version %d (want %d)", version, Version)
	}
	nTensors := c.u64()
	nKV := c.u64()
	if c.err != nil {
		return nil, fmt.Errorf("gguf: read header: %w", c.err)
	}

	f := &File{
		src:       src,
		metadata:  make(map[string]Value, nKV),
		tensors:   make(map[string]TensorInfo, nTensors),
		alignment: DefaultAlignment,
	}
	for i := uint64(0); i < nKV; i++ {
		key := c.str()
		val := readValue(c)
		if c.err != nil {
			return nil, fmt.Errorf("gguf: read metadata kv %d: %w", i, c.err)
		}
		f.metadata[key] = val
	}
	if a, ok := f.metadata[AlignmentKey]; ok {
		n, err := a.AsUint64()
		if err == nil && (n == 0 || n > maxAlignment || n&(n-1) != 0) {
			return nil, fmt.Errorf("gguf: invalid alignment %d (must be a power of two <= %d)", n, maxAlignment)
		}
		if err == nil {
			f.alignment = int64(n) // #nosec G115 -- bounded by maxAlignment above
		}
	}

	for i := uint64(0); i < nTensors; i++ {
		ti := readTensorInfo(c)
		if c.err != nil {
			return nil, fmt.Errorf("gguf: read tensor info %d: %w", i, c.err)
		}
		f.tensors[ti.Name] = ti
		f.order = append(f.order, ti.Name)
	}

	// Data section begins at the next alignment boundary after the header.
	f.dataBase = align(c.off, f.alignment)
	if f.dataBase > size {
		return nil, fmt.Errorf("gguf: data section start %d beyond file size %d", f.dataBase, size)
	}

	// Derive each tensor's on-disk byte length from the gap to the next
	// tensor offset (or the end of the data section for the last tensor).
	// This is type-agnostic and works for custom quantization encodings.
	if err := f.computeRawLengths(size - f.dataBase); err != nil {
		return nil, err
	}
	return f, nil
}

// computeRawLengths fills f.rawLen using offset boundaries within a data
// section of dataLen bytes.
func (f *File) computeRawLengths(dataLen int64) error {
	type ot struct {
		name   string
		offset int64
	}
	ots := make([]ot, 0, len(f.tensors))
	for name, ti := range f.tensors {
		ots = append(ots, ot{name, int64(ti.Offset)}) // #nosec G115 -- bounded by maxTensorOffset in readTensorInfo
	}
	sort.Slice(ots, func(i, j int) bool { return ots[i].offset < ots[j].offset })

	f.rawLen = make(map[string]int64, len(ots))
	for i, cur := range ots {
		end := dataLen
		if i+1 < len(ots) {
			end = ots[i+1].offset
		}
		if cur.offset < 0 || end < cur.offset || end > dataLen {
			return fmt.Errorf("gguf: tensor %q has invalid byte range [%d, %d]", cur.name, cur.offset, end)
		}
		f.rawLen[cur.name] = end - cur.offset
	}
	return nil
}

func readValue(c *cursor) Value {
	t := metadataValueType(c.u32())
	return readValueOfType(c, t)
}

func readValueOfType(c *cursor, t metadataValueType) Value {
	switch t {
	case mvUint8, mvInt8:
		b := c.read(1)
		if b == nil {
			return Value{Type: t}
		}
		return Value{Type: t, Num: uint64(b[0])}
	case mvUint16, mvInt16:
		b := c.read(2)
		if b == nil {
			return Value{Type: t}
		}
		return Value{Type: t, Num: uint64(binary.LittleEndian.Uint16(b))}
	case mvUint32, mvInt32:
		return Value{Type: t, Num: uint64(c.u32())}
	case mvFloat32:
		return Value{Type: t, F64: float64(f32FromBits(c.u32()))}
	case mvBool:
		b := c.read(1)
		if b == nil {
			return Value{Type: t}
		}
		return Value{Type: t, Num: uint64(b[0])}
	case mvString:
		return Value{Type: t, Str: c.str()}
	case mvUint64, mvInt64:
		return Value{Type: t, Num: c.u64()}
	case mvFloat64:
		return Value{Type: t, F64: f64FromBits(c.u64())}
	case mvArray:
		elemType := metadataValueType(c.u32())
		n := c.u64()
		if c.err != nil {
			return Value{Type: t}
		}
		arr := make([]Value, 0, n)
		for i := uint64(0); i < n && c.err == nil; i++ {
			arr = append(arr, readValueOfType(c, elemType))
		}
		return Value{Type: mvArray, ArrayType: elemType, Array: arr}
	default:
		c.err = fmt.Errorf("gguf: unknown metadata value type %d", t)
		return Value{Type: t}
	}
}

func readTensorInfo(c *cursor) TensorInfo {
	name := c.str()
	nDims := c.u32()
	if c.err != nil {
		return TensorInfo{}
	}
	if nDims > 8 {
		c.err = fmt.Errorf("gguf: tensor %q has implausible n_dims %d", name, nDims)
		return TensorInfo{}
	}
	dims := make([]uint64, nDims)
	for i := range dims {
		dims[i] = c.u64()
	}
	typ := GGMLType(c.u32())
	off := c.u64()
	elems := uint64(1)
	for _, d := range dims {
		if d != 0 && elems > maxTensorElems/d {
			c.err = fmt.Errorf("gguf: tensor %q element count exceeds %d", name, uint64(maxTensorElems))
			return TensorInfo{}
		}
		elems *= d
	}
	if off > maxTensorOffset {
		c.err = fmt.Errorf("gguf: tensor %q offset %d exceeds %d", name, off, uint64(maxTensorOffset))
		return TensorInfo{}
	}
	return TensorInfo{Name: name, Dims: dims, Type: typ, Offset: off}
}

// align rounds off up to the next multiple of a.
func align(off, a int64) int64 {
	if a <= 1 {
		return off
	}
	if rem := off % a; rem != 0 {
		return off + (a - rem)
	}
	return off
}

// Names returns tensor names in file order.
func (f *File) Names() []string {
	out := append([]string(nil), f.order...)
	return out
}

// MetadataKeys returns the metadata keys sorted lexically.
func (f *File) MetadataKeys() []string {
	keys := make([]string, 0, len(f.metadata))
	for k := range f.metadata {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// Metadata returns the value for key.
func (f *File) Metadata(key string) (Value, bool) {
	v, ok := f.metadata[key]
	return v, ok
}

// Alignment returns the resolved tensor-data alignment.
func (f *File) Alignment() int64 { return f.alignment }

// Info returns the TensorInfo for name.
func (f *File) Info(name string) (TensorInfo, bool) {
	ti, ok := f.tensors[name]
	return ti, ok
}

// rawBytes returns the on-disk byte length of a tensor. For types with a known
// block layout it computes the exact size (and validates it fits the boundary
// gap); for custom or unmodeled types it falls back to the offset-derived
// boundary length.
func (f *File) rawBytes(ti TensorInfo) (int64, error) {
	boundary := f.rawLen[ti.Name]
	spec, ok := blockSpecs[ti.Type]
	if !ok {
		return boundary, nil
	}
	n := ti.numElements()
	if n%uint64(spec.elemsPerBlock) != 0 { // #nosec G115 -- block specs are small positive constants
		return 0, fmt.Errorf("gguf: tensor %q element count %d not a multiple of block %d",
			ti.Name, n, spec.elemsPerBlock)
	}
	// n is bounded by maxTensorElems and block specs are small constants,
	// so this product stays inside int64.
	exact := int64(n/uint64(spec.elemsPerBlock)) * int64(spec.bytesPerBlock) // #nosec G115
	if exact > boundary {
		return 0, fmt.Errorf("gguf: tensor %q needs %d bytes but only %d available before next tensor",
			ti.Name, exact, boundary)
	}
	return exact, nil
}

// Raw reads the raw on-disk bytes for the named tensor.
func (f *File) Raw(name string) ([]byte, error) {
	ti, ok := f.tensors[name]
	if !ok {
		return nil, fmt.Errorf("gguf: tensor %q not found", name)
	}
	nbytes, err := f.rawBytes(ti)
	if err != nil {
		return nil, err
	}
	buf := make([]byte, nbytes)
	if _, err := f.src.ReadAt(buf, f.dataBase+int64(ti.Offset)); err != nil { // #nosec G115 -- bounded by maxTensorOffset in readTensorInfo
		return nil, fmt.Errorf("gguf: read tensor %q: %w", name, err)
	}
	return buf, nil
}

// Float32 reads and dequantizes the named tensor to float32.
func (f *File) Float32(name string) ([]float32, error) {
	ti, ok := f.tensors[name]
	if !ok {
		return nil, fmt.Errorf("gguf: tensor %q not found", name)
	}
	raw, err := f.Raw(name)
	if err != nil {
		return nil, err
	}
	return dequantize(ti.Type, raw, int(ti.numElements())) // #nosec G115 -- bounded by maxTensorElems in readTensorInfo
}

// Close releases the file handle if File was created via Open.
func (f *File) Close() error {
	if f.closer != nil {
		return f.closer.Close()
	}
	return nil
}
