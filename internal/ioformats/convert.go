package ioformats

import (
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/gabrielleeyj/turbodb/pkg/formats/gguf"
	"github.com/gabrielleeyj/turbodb/pkg/formats/safetensors"
)

// TensorSummary is a format-agnostic description of one tensor, used by inspect.
type TensorSummary struct {
	Name  string
	Dtype string
	Shape []int64
}

// Inspect returns a sorted summary of the tensors and metadata in a file.
func Inspect(format Format, path string) (tensors []TensorSummary, metadata map[string]string, err error) {
	switch format {
	case FormatSafeTensors:
		f, oerr := safetensors.Open(path)
		if oerr != nil {
			return nil, nil, oerr
		}
		defer f.Close()
		for _, name := range f.Names() {
			info, _ := f.Info(name)
			tensors = append(tensors, TensorSummary{Name: name, Dtype: string(info.Dtype), Shape: info.Shape})
		}
		return tensors, f.Metadata(), nil
	case FormatGGUF:
		f, oerr := gguf.Open(path)
		if oerr != nil {
			return nil, nil, oerr
		}
		defer f.Close()
		for _, name := range f.Names() {
			info, _ := f.Info(name)
			shape := make([]int64, len(info.Dims))
			for i, d := range info.Dims {
				shape[i] = int64(d)
			}
			tensors = append(tensors, TensorSummary{Name: name, Dtype: info.Type.String(), Shape: shape})
		}
		sort.Slice(tensors, func(i, j int) bool { return tensors[i].Name < tensors[j].Name })
		md := map[string]string{}
		for _, k := range f.MetadataKeys() {
			v, _ := f.Metadata(k)
			if s, e := v.AsString(); e == nil {
				md[k] = s
			}
		}
		return tensors, md, nil
	default:
		return nil, nil, fmt.Errorf("ioformats: unsupported format %q", format)
	}
}

// ConvertSafeTensorsToGGUF reads every tensor from a SafeTensors file and writes
// them to a GGUF file. Float tensors are stored as F32; other dtypes are
// preserved as raw bytes under their nearest GGUF type where possible.
func ConvertSafeTensorsToGGUF(in string, out io.Writer) error {
	f, err := safetensors.Open(in)
	if err != nil {
		return err
	}
	defer f.Close()

	w := gguf.NewWriter(out)
	w.SetMetadata("general.architecture", gguf.StringValue("turboquant-export"))
	for k, v := range f.Metadata() {
		w.SetMetadata("safetensors."+k, gguf.StringValue(v))
	}

	names := f.Names()
	payloads := make(map[string][]byte, len(names))
	for _, name := range names {
		info, _ := f.Info(name)
		tn, terr := f.Tensor(name)
		if terr != nil {
			return terr
		}
		dims := reverseDims(info.Shape) // ggml ne[0] is fastest-varying
		if err := w.Declare(ggufName(name), safetensorsToGGML(info.Dtype), dims, int64(len(tn.Data))); err != nil {
			return err
		}
		payloads[name] = tn.Data
	}
	if err := w.Commit(); err != nil {
		return err
	}
	for _, name := range names {
		if err := w.WriteData(ggufName(name), payloads[name]); err != nil {
			return err
		}
	}
	return w.Close()
}

// safetensorsToGGML maps a SafeTensors dtype to the closest GGUF type for raw
// byte preservation.
func safetensorsToGGML(d safetensors.Dtype) gguf.GGMLType {
	switch d {
	case safetensors.F16:
		return gguf.GGMLTypeF16
	default:
		return gguf.GGMLTypeF32
	}
}

func reverseDims(shape []int64) []uint64 {
	out := make([]uint64, len(shape))
	for i, d := range shape {
		out[len(shape)-1-i] = uint64(d)
	}
	return out
}

// ggufName sanitizes a tensor name (GGUF names are arbitrary strings, but we
// trim surrounding whitespace for cleanliness).
func ggufName(name string) string { return strings.TrimSpace(name) }
