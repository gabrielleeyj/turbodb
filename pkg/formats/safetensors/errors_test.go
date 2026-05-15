package safetensors

import (
	"bytes"
	"testing"
)

func TestDeclareErrors(t *testing.T) {
	w := NewWriter(&bytes.Buffer{}, nil)
	if err := w.Declare("bad", "NOPE", []int64{1}); err == nil {
		t.Error("expected unknown dtype error")
	}
	if err := w.Declare("neg", F32, []int64{-1}); err == nil {
		t.Error("expected negative-dim error")
	}
	if err := w.Declare("a", F32, []int64{1}); err != nil {
		t.Fatal(err)
	}
	if err := w.Declare("a", F32, []int64{1}); err == nil {
		t.Error("expected duplicate-name error")
	}
	if err := w.Commit(); err != nil {
		t.Fatal(err)
	}
	if err := w.Declare("late", F32, []int64{1}); err == nil {
		t.Error("expected declare-after-commit error")
	}
	if err := w.Commit(); err == nil {
		t.Error("expected double-commit error")
	}
}

func TestWriteBeforeCommit(t *testing.T) {
	w := NewWriter(&bytes.Buffer{}, nil)
	if err := w.Write("x", []byte{1}); err == nil {
		t.Error("expected write-before-commit error")
	}
	if err := w.Close(); err == nil {
		t.Error("expected close-before-commit error")
	}
}

func TestWriteWrongSize(t *testing.T) {
	w := NewWriter(&bytes.Buffer{}, nil)
	_ = w.Declare("a", U8, []int64{3})
	_ = w.Commit()
	if err := w.Write("a", []byte{1, 2}); err == nil {
		t.Error("expected size-mismatch error")
	}
}

func TestParseQuantMetaErrors(t *testing.T) {
	if _, err := ParseQuantMeta(map[string]string{MetaRotatorSeed: "notanumber"}); err == nil {
		t.Error("expected bad seed error")
	}
	if _, err := ParseQuantMeta(map[string]string{MetaBitWidth: "x"}); err == nil {
		t.Error("expected bad bit-width error")
	}
	m, err := ParseQuantMeta(map[string]string{})
	if err != nil || m.BitWidth != 0 {
		t.Errorf("empty metadata should parse to zero value, got %+v err %v", m, err)
	}
}

func TestTensorFloat32NonFloat(t *testing.T) {
	tn := &Tensor{Name: "x", Info: TensorInfo{Dtype: U8, Shape: []int64{2}}, Data: []byte{1, 2}}
	if _, err := tn.Float32(); err == nil {
		t.Error("expected non-float dtype error")
	}
}

func TestTensorNotFound(t *testing.T) {
	var buf bytes.Buffer
	if err := Save(&buf, []*Tensor{{Name: "a", Info: TensorInfo{Dtype: U8, Shape: []int64{1}}, Data: []byte{1}}}, nil); err != nil {
		t.Fatal(err)
	}
	f, err := NewReader(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := f.Info("missing"); ok {
		t.Error("Info should report missing tensor")
	}
	if _, err := f.Tensor("missing"); err == nil {
		t.Error("expected not-found error")
	}
	if f.Metadata() != nil {
		t.Error("expected nil metadata")
	}
	if err := f.Close(); err != nil {
		t.Error(err)
	}
}

func TestFloat16SpecialValues(t *testing.T) {
	// Inf, -Inf, zero, subnormal-ish small values exercise the special paths.
	cases := []uint16{0x0000, 0x8000, 0x7c00, 0xfc00, 0x0001, 0x03ff}
	for _, h := range cases {
		_ = float16ToFloat32(h) // must not panic; value correctness covered elsewhere
	}
	// Round-trip of large/small magnitudes through the float32->float16 path.
	for _, f := range []float32{0, 1e-9, 70000, -70000, 1e30} {
		_ = float32ToFloat16(f)
	}
}

func TestNonContiguousHeaderRejected(t *testing.T) {
	// Gap between tensor a (ends at 1) and b (starts at 2).
	bad := `{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[2,3]}}`
	if _, err := parseHeader([]byte(bad), 3); err == nil {
		t.Error("expected non-contiguous layout error")
	}
}

func TestParseHeaderRejectsBadJSON(t *testing.T) {
	if _, err := parseHeader([]byte("{not json"), 0); err == nil {
		t.Error("expected json parse error")
	}
	// Tensor extends beyond data section.
	bad := `{"t":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}`
	if _, err := parseHeader([]byte(bad), 2); err == nil {
		t.Error("expected out-of-range error")
	}
}
