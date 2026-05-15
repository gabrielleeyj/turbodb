// Package gguf implements a pure-Go reader and writer for the GGUF tensor file
// format (version 3) used by llama.cpp/ggml.
//
// Layout:
//
//	magic     uint32  // "GGUF" little-endian
//	version   uint32  // 3
//	n_tensors uint64
//	n_kv      uint64
//	metadata  [n_kv]  // key/value pairs (see Value)
//	tensors   [n_tensors] tensorInfo
//	padding             // to general.alignment (default 32)
//	data                // tensor bytes at info.Offset within this section
package gguf

// Magic is the 4-byte file magic "GGUF" interpreted as a little-endian uint32.
const Magic uint32 = 0x46554747 // 'G','G','U','F'

// Version is the GGUF format version this package reads and writes.
const Version uint32 = 3

// DefaultAlignment is the tensor-data alignment used when the
// general.alignment metadata key is absent.
const DefaultAlignment = 32

// AlignmentKey is the metadata key controlling tensor-data alignment.
const AlignmentKey = "general.alignment"

// GGMLType identifies the on-disk encoding of a tensor's elements.
type GGMLType uint32

const (
	GGMLTypeF32  GGMLType = 0
	GGMLTypeF16  GGMLType = 1
	GGMLTypeQ4_0 GGMLType = 2
	GGMLTypeQ4_1 GGMLType = 3
	GGMLTypeQ5_0 GGMLType = 6
	GGMLTypeQ5_1 GGMLType = 7
	GGMLTypeQ8_0 GGMLType = 8
	GGMLTypeQ8_1 GGMLType = 9
	GGMLTypeQ4_K GGMLType = 12
	GGMLTypeQ6_K GGMLType = 14

	// TurboQuant custom types, allocated in a high namespace so they never
	// clash with future upstream ggml types.
	GGMLTypeTurboQuantMSE  GGMLType = 128
	GGMLTypeTurboQuantProd GGMLType = 129
)

// blockSpec describes the block layout of a quantized ggml type: how many
// elements share a block and how many bytes that block occupies.
type blockSpec struct {
	elemsPerBlock int
	bytesPerBlock int
}

// blockSpecs maps the ggml types this package can dequantize to their block
// geometry. Types absent here are parsed but not dequantizable.
var blockSpecs = map[GGMLType]blockSpec{
	GGMLTypeF32:  {1, 4},
	GGMLTypeF16:  {1, 2},
	GGMLTypeQ8_0: {32, 34}, // f16 scale + 32 int8
	GGMLTypeQ4_0: {32, 18}, // f16 scale + 16 packed nibbles
	GGMLTypeQ4_1: {32, 20}, // f16 scale + f16 min + 16 packed nibbles
}

// String renders a human-readable type name.
func (t GGMLType) String() string {
	switch t {
	case GGMLTypeF32:
		return "F32"
	case GGMLTypeF16:
		return "F16"
	case GGMLTypeQ4_0:
		return "Q4_0"
	case GGMLTypeQ4_1:
		return "Q4_1"
	case GGMLTypeQ5_0:
		return "Q5_0"
	case GGMLTypeQ5_1:
		return "Q5_1"
	case GGMLTypeQ8_0:
		return "Q8_0"
	case GGMLTypeQ8_1:
		return "Q8_1"
	case GGMLTypeQ4_K:
		return "Q4_K"
	case GGMLTypeQ6_K:
		return "Q6_K"
	case GGMLTypeTurboQuantMSE:
		return "TURBOQUANT_MSE"
	case GGMLTypeTurboQuantProd:
		return "TURBOQUANT_PROD"
	default:
		return "UNKNOWN"
	}
}

// metadataValueType enumerates GGUF metadata value kinds.
type metadataValueType uint32

const (
	mvUint8   metadataValueType = 0
	mvInt8    metadataValueType = 1
	mvUint16  metadataValueType = 2
	mvInt16   metadataValueType = 3
	mvUint32  metadataValueType = 4
	mvInt32   metadataValueType = 5
	mvFloat32 metadataValueType = 6
	mvBool    metadataValueType = 7
	mvString  metadataValueType = 8
	mvArray   metadataValueType = 9
	mvUint64  metadataValueType = 10
	mvInt64   metadataValueType = 11
	mvFloat64 metadataValueType = 12
)
