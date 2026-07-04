// Package ioformats bridges the on-disk tensor formats (SafeTensors, GGUF) and
// the TurboDB engine, implementing the import/export operations exposed by
// turbodb-ctl.
package ioformats

import (
	"fmt"

	"github.com/gabrielleeyj/turbodb/pkg/formats/gguf"
	"github.com/gabrielleeyj/turbodb/pkg/formats/safetensors"
)

// Format identifies a supported tensor file format.
type Format string

// Supported tensor file formats.
const (
	FormatSafeTensors Format = "safetensors"
	FormatGGUF        Format = "gguf"
)

// Matrix is a dense row-major float32 matrix of Rows vectors each of length Dim.
type Matrix struct {
	Rows   int
	Dim    int
	Values []float32 // length Rows*Dim, row-major
}

// Row returns a view of row i (no copy).
func (m Matrix) Row(i int) []float32 {
	return m.Values[i*m.Dim : (i+1)*m.Dim]
}

// ReadMatrix loads a 2D float tensor from a file as a row-major Matrix.
// tensorName selects the tensor; for SafeTensors an empty name selects the sole
// tensor (error if ambiguous). For GGUF a name is required.
func ReadMatrix(format Format, path, tensorName string) (Matrix, error) {
	switch format {
	case FormatSafeTensors:
		return readSafeTensorsMatrix(path, tensorName)
	case FormatGGUF:
		return readGGUFMatrix(path, tensorName)
	default:
		return Matrix{}, fmt.Errorf("ioformats: unsupported format %q", format)
	}
}

func readSafeTensorsMatrix(path, tensorName string) (Matrix, error) {
	f, err := safetensors.Open(path)
	if err != nil {
		return Matrix{}, err
	}
	defer func() { _ = f.Close() }()

	name := tensorName
	if name == "" {
		names := f.Names()
		if len(names) != 1 {
			return Matrix{}, fmt.Errorf("ioformats: %d tensors in %q; specify --tensor", len(names), path)
		}
		name = names[0]
	}
	info, ok := f.Info(name)
	if !ok {
		return Matrix{}, fmt.Errorf("ioformats: tensor %q not found in %q", name, path)
	}
	rows, dim, err := shape2D(info.Shape)
	if err != nil {
		return Matrix{}, fmt.Errorf("ioformats: tensor %q: %w", name, err)
	}
	tn, err := f.Tensor(name)
	if err != nil {
		return Matrix{}, err
	}
	values, err := tn.Float32()
	if err != nil {
		return Matrix{}, err
	}
	return Matrix{Rows: rows, Dim: dim, Values: values}, nil
}

func readGGUFMatrix(path, tensorName string) (Matrix, error) {
	if tensorName == "" {
		return Matrix{}, fmt.Errorf("ioformats: GGUF import requires --tensor")
	}
	f, err := gguf.Open(path)
	if err != nil {
		return Matrix{}, err
	}
	defer func() { _ = f.Close() }()

	info, ok := f.Info(tensorName)
	if !ok {
		return Matrix{}, fmt.Errorf("ioformats: tensor %q not found in %q", tensorName, path)
	}
	// ggml stores ne[0] as the fastest-varying dimension (the vector width);
	// the remaining dimensions multiply into the row count.
	if len(info.Dims) == 0 {
		return Matrix{}, fmt.Errorf("ioformats: tensor %q has no dimensions", tensorName)
	}
	dim := int(info.Dims[0]) // #nosec G115 -- gguf reader bounds element counts to 2^48
	total := 1
	for _, d := range info.Dims {
		total *= int(d) // #nosec G115 -- gguf reader bounds element counts to 2^48
	}
	if dim <= 0 || total%dim != 0 {
		return Matrix{}, fmt.Errorf("ioformats: tensor %q dims %v not a 2D matrix", tensorName, info.Dims)
	}
	values, err := f.Float32(tensorName)
	if err != nil {
		return Matrix{}, err
	}
	return Matrix{Rows: total / dim, Dim: dim, Values: values}, nil
}

// shape2D interprets a tensor shape as [rows, dim]. A 1D shape [d] is treated as
// a single row.
func shape2D(shape []int64) (rows, dim int, err error) {
	switch len(shape) {
	case 1:
		return 1, int(shape[0]), nil
	case 2:
		return int(shape[0]), int(shape[1]), nil
	default:
		return 0, 0, fmt.Errorf("expected 1D or 2D shape, got %v", shape)
	}
}
