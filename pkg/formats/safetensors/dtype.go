// Package safetensors implements a pure-Go reader and writer for the
// SafeTensors file format (https://github.com/huggingface/safetensors).
//
// The format is a length-prefixed JSON header followed by a contiguous tensor
// data section:
//
//	[8 bytes little-endian uint64 header length N]
//	[N bytes UTF-8 JSON header]
//	[tensor data section]
//
// The JSON header maps each tensor name to its dtype, shape, and a
// [begin, end) byte range within the data section. A reserved "__metadata__"
// key carries free-form string metadata, which TurboDB uses to record
// quantization parameters (see Metadata).
package safetensors

import "fmt"

// Dtype enumerates the tensor element types defined by the SafeTensors spec.
type Dtype string

const (
	F64  Dtype = "F64"
	F32  Dtype = "F32"
	F16  Dtype = "F16"
	BF16 Dtype = "BF16"
	I64  Dtype = "I64"
	I32  Dtype = "I32"
	I16  Dtype = "I16"
	I8   Dtype = "I8"
	U8   Dtype = "U8"
	BOOL Dtype = "BOOL"
)

// MaxHeaderBytes is the maximum allowed JSON header size per the SafeTensors
// spec (100 MB). Headers larger than this are rejected to bound memory use and
// guard against malformed or hostile files.
const MaxHeaderBytes = 100 * 1024 * 1024

// ByteSize returns the number of bytes occupied by a single element of the
// dtype. It returns an error for unknown dtypes so callers can fail fast on
// untrusted input rather than computing bogus byte ranges.
func (d Dtype) ByteSize() (int, error) {
	switch d {
	case F64, I64:
		return 8, nil
	case F32, I32:
		return 4, nil
	case F16, BF16, I16:
		return 2, nil
	case I8, U8, BOOL:
		return 1, nil
	default:
		return 0, fmt.Errorf("safetensors: unknown dtype %q", d)
	}
}

// IsKnown reports whether d is a dtype this package understands.
func (d Dtype) IsKnown() bool {
	_, err := d.ByteSize()
	return err == nil
}

// numElements returns the product of the shape dimensions, validating that no
// dimension is negative and that the product does not overflow. An empty shape
// denotes a scalar (one element).
func numElements(shape []int64) (int64, error) {
	count := int64(1)
	for _, dim := range shape {
		if dim < 0 {
			return 0, fmt.Errorf("safetensors: negative dimension %d in shape %v", dim, shape)
		}
		if dim != 0 && count > maxInt64/dim {
			return 0, fmt.Errorf("safetensors: element count overflow for shape %v", shape)
		}
		count *= dim
	}
	return count, nil
}

const maxInt64 = int64(^uint64(0) >> 1)
