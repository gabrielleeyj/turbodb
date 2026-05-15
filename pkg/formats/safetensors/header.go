package safetensors

import (
	"encoding/json"
	"fmt"
	"sort"
)

// metadataKey is the reserved JSON key carrying free-form string metadata.
const metadataKey = "__metadata__"

// TensorInfo describes the location and type of one tensor within the data
// section. DataOffsets is a [begin, end) byte range relative to the start of
// the data section.
type TensorInfo struct {
	Dtype       Dtype    `json:"dtype"`
	Shape       []int64  `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// nbytes returns the byte length implied by the offsets.
func (t TensorInfo) nbytes() int64 { return t.DataOffsets[1] - t.DataOffsets[0] }

// validate checks that the dtype is known and that the declared byte range is
// consistent with the dtype and shape.
func (t TensorInfo) validate(name string) error {
	elemSize, err := t.Dtype.ByteSize()
	if err != nil {
		return fmt.Errorf("tensor %q: %w", name, err)
	}
	count, err := numElements(t.Shape)
	if err != nil {
		return fmt.Errorf("tensor %q: %w", name, err)
	}
	begin, end := t.DataOffsets[0], t.DataOffsets[1]
	if begin < 0 || end < begin {
		return fmt.Errorf("tensor %q: invalid data_offsets [%d, %d]", name, begin, end)
	}
	if want := count * int64(elemSize); t.nbytes() != want {
		return fmt.Errorf("tensor %q: data range %d bytes does not match shape %v dtype %s (%d bytes)",
			name, t.nbytes(), t.Shape, t.Dtype, want)
	}
	return nil
}

// header is the in-memory representation of the parsed JSON header.
type header struct {
	tensors  map[string]TensorInfo
	metadata map[string]string
}

// parseHeader decodes and validates the JSON header bytes. It enforces that
// every tensor's byte range stays within dataLen and that ranges are
// contiguous and non-overlapping when sorted by start offset.
func parseHeader(raw []byte, dataLen int64) (*header, error) {
	// Decode into a generic map first so we can split out the reserved
	// metadata key from the tensor entries.
	var rawMap map[string]json.RawMessage
	if err := json.Unmarshal(raw, &rawMap); err != nil {
		return nil, fmt.Errorf("safetensors: parse header json: %w", err)
	}

	h := &header{tensors: make(map[string]TensorInfo, len(rawMap))}
	for name, msg := range rawMap {
		if name == metadataKey {
			if err := json.Unmarshal(msg, &h.metadata); err != nil {
				return nil, fmt.Errorf("safetensors: parse __metadata__: %w", err)
			}
			continue
		}
		var info TensorInfo
		if err := json.Unmarshal(msg, &info); err != nil {
			return nil, fmt.Errorf("safetensors: parse tensor %q: %w", name, err)
		}
		if err := info.validate(name); err != nil {
			return nil, fmt.Errorf("safetensors: %w", err)
		}
		if info.DataOffsets[1] > dataLen {
			return nil, fmt.Errorf("safetensors: tensor %q ends at %d beyond data section of %d bytes",
				name, info.DataOffsets[1], dataLen)
		}
		h.tensors[name] = info
	}
	if err := h.checkContiguous(); err != nil {
		return nil, err
	}
	return h, nil
}

// checkContiguous verifies the tensor byte ranges tile the data section with no
// gaps or overlaps, which the SafeTensors validators require.
func (h *header) checkContiguous() error {
	infos := make([]TensorInfo, 0, len(h.tensors))
	for _, info := range h.tensors {
		infos = append(infos, info)
	}
	sort.Slice(infos, func(i, j int) bool {
		return infos[i].DataOffsets[0] < infos[j].DataOffsets[0]
	})
	var cursor int64
	for _, info := range infos {
		if info.DataOffsets[0] != cursor {
			return fmt.Errorf("safetensors: non-contiguous tensor layout: expected offset %d, got %d",
				cursor, info.DataOffsets[0])
		}
		cursor = info.DataOffsets[1]
	}
	return nil
}

// buildHeaderJSON serializes tensor infos and metadata into the canonical
// header byte representation. Tensor names are emitted in sorted order for
// deterministic, reproducible output.
func buildHeaderJSON(tensors map[string]TensorInfo, metadata map[string]string) ([]byte, error) {
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	sort.Strings(names)

	// Use an ordered slice of key/value json fragments to keep output stable.
	obj := make(map[string]any, len(tensors)+1)
	if len(metadata) > 0 {
		obj[metadataKey] = metadata
	}
	for _, name := range names {
		obj[name] = tensors[name]
	}
	out, err := json.Marshal(obj)
	if err != nil {
		return nil, fmt.Errorf("safetensors: marshal header: %w", err)
	}
	if len(out) > MaxHeaderBytes {
		return nil, fmt.Errorf("safetensors: header %d bytes exceeds max %d", len(out), MaxHeaderBytes)
	}
	return out, nil
}
