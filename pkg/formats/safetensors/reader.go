package safetensors

import (
	"encoding/binary"
	"fmt"
	"io"
	"iter"
	"math"
	"sort"

	"golang.org/x/exp/mmap"
)

// headerLenSize is the fixed 8-byte little-endian length prefix.
const headerLenSize = 8

// File provides read access to a SafeTensors file. It is backed by an
// io.ReaderAt so it works equally over an mmap'd file (zero-copy backing) or
// any other random-access source. File is safe for concurrent reads.
type File struct {
	src      io.ReaderAt
	closer   io.Closer
	hdr      *header
	dataBase int64 // byte offset where the data section starts
}

// Open memory-maps the SafeTensors file at path for zero-copy backing and
// parses its header. The returned File must be closed to release the mapping.
func Open(path string) (*File, error) {
	r, err := mmap.Open(path)
	if err != nil {
		return nil, fmt.Errorf("safetensors: open %q: %w", path, err)
	}
	f, err := NewReader(r, int64(r.Len()))
	if err != nil {
		_ = r.Close()
		return nil, err
	}
	f.closer = r
	return f, nil
}

// NewReader parses the SafeTensors header from src, which must expose exactly
// size bytes. src is retained for lazy tensor reads.
func NewReader(src io.ReaderAt, size int64) (*File, error) {
	if size < headerLenSize {
		return nil, fmt.Errorf("safetensors: file too small (%d bytes) for header length prefix", size)
	}
	var lenBuf [headerLenSize]byte
	if _, err := src.ReadAt(lenBuf[:], 0); err != nil {
		return nil, fmt.Errorf("safetensors: read header length: %w", err)
	}
	headerLen := binary.LittleEndian.Uint64(lenBuf[:])
	if headerLen > MaxHeaderBytes {
		return nil, fmt.Errorf("safetensors: header length %d exceeds max %d", headerLen, MaxHeaderBytes)
	}
	dataBase := headerLenSize + int64(headerLen)
	if dataBase > size {
		return nil, fmt.Errorf("safetensors: header length %d exceeds file size %d", headerLen, size)
	}

	rawHeader := make([]byte, headerLen)
	if _, err := src.ReadAt(rawHeader, headerLenSize); err != nil {
		return nil, fmt.Errorf("safetensors: read header: %w", err)
	}
	hdr, err := parseHeader(rawHeader, size-dataBase)
	if err != nil {
		return nil, err
	}
	return &File{src: src, hdr: hdr, dataBase: dataBase}, nil
}

// Names returns the tensor names sorted lexically.
func (f *File) Names() []string {
	names := make([]string, 0, len(f.hdr.tensors))
	for name := range f.hdr.tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// Metadata returns a copy of the free-form __metadata__ map (nil if absent).
func (f *File) Metadata() map[string]string {
	if f.hdr.metadata == nil {
		return nil
	}
	out := make(map[string]string, len(f.hdr.metadata))
	for k, v := range f.hdr.metadata {
		out[k] = v
	}
	return out
}

// Info returns the TensorInfo for name.
func (f *File) Info(name string) (TensorInfo, bool) {
	info, ok := f.hdr.tensors[name]
	return info, ok
}

// Tensor loads the raw bytes for the named tensor. The returned Tensor owns its
// byte slice (a copy of the mapped region), so callers may retain it after the
// File is closed.
func (f *File) Tensor(name string) (*Tensor, error) {
	info, ok := f.hdr.tensors[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: tensor %q not found", name)
	}
	buf := make([]byte, info.nbytes())
	if _, err := f.src.ReadAt(buf, f.dataBase+info.DataOffsets[0]); err != nil {
		return nil, fmt.Errorf("safetensors: read tensor %q: %w", name, err)
	}
	return &Tensor{Name: name, Info: info, Data: buf}, nil
}

// Iter yields every tensor in sorted name order, loading each lazily. Using a
// range-over-func iterator avoids materializing all tensors at once, which
// matters for multi-gigabyte files. Iteration stops early if yield returns
// false; the first load error is reported via the returned *error pointer.
func (f *File) Iter() (iter.Seq2[string, *Tensor], *error) {
	var iterErr error
	seq := func(yield func(string, *Tensor) bool) {
		for _, name := range f.Names() {
			t, err := f.Tensor(name)
			if err != nil {
				iterErr = err
				return
			}
			if !yield(name, t) {
				return
			}
		}
	}
	return seq, &iterErr
}

// Close releases the underlying mapping if File was created via Open.
func (f *File) Close() error {
	if f.closer != nil {
		return f.closer.Close()
	}
	return nil
}

// Tensor is a loaded tensor: its metadata plus raw little-endian bytes.
type Tensor struct {
	Name string
	Info TensorInfo
	Data []byte
}

// Float32 decodes the tensor into a freshly allocated []float32, converting
// F16/BF16/F32 source data. It errors for non-float dtypes.
func (t *Tensor) Float32() ([]float32, error) {
	elemSize, _ := t.Info.Dtype.ByteSize()
	n := len(t.Data) / elemSize
	out := make([]float32, n)
	switch t.Info.Dtype {
	case F32:
		for i := 0; i < n; i++ {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(t.Data[i*4:]))
		}
	case F16:
		for i := 0; i < n; i++ {
			out[i] = float16ToFloat32(binary.LittleEndian.Uint16(t.Data[i*2:]))
		}
	case BF16:
		for i := 0; i < n; i++ {
			out[i] = bfloat16ToFloat32(binary.LittleEndian.Uint16(t.Data[i*2:]))
		}
	default:
		return nil, fmt.Errorf("safetensors: tensor %q dtype %s is not float-convertible", t.Name, t.Info.Dtype)
	}
	return out, nil
}
