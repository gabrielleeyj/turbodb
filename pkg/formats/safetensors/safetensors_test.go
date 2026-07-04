package safetensors

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// f32Bytes encodes float32 values as little-endian bytes.
func f32Bytes(vals ...float32) []byte {
	buf := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

func TestDtypeByteSize(t *testing.T) {
	cases := []struct {
		dtype Dtype
		want  int
		ok    bool
	}{
		{F32, 4, true},
		{F16, 2, true},
		{BF16, 2, true},
		{I64, 8, true},
		{U8, 1, true},
		{BOOL, 1, true},
		{"NOPE", 0, false},
	}
	for _, c := range cases {
		got, err := c.dtype.ByteSize()
		if (err == nil) != c.ok {
			t.Errorf("dtype %s: ok=%v, err=%v", c.dtype, c.ok, err)
		}
		if got != c.want {
			t.Errorf("dtype %s: size=%d want %d", c.dtype, got, c.want)
		}
	}
}

func TestRoundTripInMemory(t *testing.T) {
	// Arrange
	tensors := []*Tensor{
		{Name: "codes", Info: TensorInfo{Dtype: F32, Shape: []int64{2, 2}}, Data: f32Bytes(1, 2, 3, 4)},
		{Name: "norms", Info: TensorInfo{Dtype: F32, Shape: []int64{2}}, Data: f32Bytes(5, 6)},
	}
	meta := QuantMeta{RotatorSeed: 17384920123, RotatorType: "hadamard", CodebookID: "d4_b4", BitWidth: 4, Variant: "mse"}

	// Act
	var buf bytes.Buffer
	if err := Save(&buf, tensors, meta.ToMap()); err != nil {
		t.Fatalf("Save: %v", err)
	}
	f, err := NewReader(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}

	// Assert
	if names := f.Names(); len(names) != 2 || names[0] != "codes" || names[1] != "norms" {
		t.Fatalf("Names = %v", names)
	}
	codes, err := f.Tensor("codes")
	if err != nil {
		t.Fatalf("Tensor: %v", err)
	}
	if !bytes.Equal(codes.Data, f32Bytes(1, 2, 3, 4)) {
		t.Errorf("codes data mismatch: %v", codes.Data)
	}
	gotMeta, err := ParseQuantMeta(f.Metadata())
	if err != nil {
		t.Fatalf("ParseQuantMeta: %v", err)
	}
	if gotMeta.RotatorSeed != 17384920123 || gotMeta.BitWidth != 4 || gotMeta.Variant != "mse" {
		t.Errorf("metadata round-trip mismatch: %+v", gotMeta)
	}
}

func TestFloat32Conversion(t *testing.T) {
	vals := []float32{0, 1, -1, 0.5, 3.5, -2.25}
	// F16 round-trip (exactly representable values chosen).
	f16 := make([]byte, 2*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint16(f16[i*2:], float32ToFloat16(v))
	}
	tn := &Tensor{Name: "t", Info: TensorInfo{Dtype: F16, Shape: []int64{int64(len(vals))}}, Data: f16}
	got, err := tn.Float32()
	if err != nil {
		t.Fatalf("Float32: %v", err)
	}
	for i, v := range vals {
		if got[i] != v {
			t.Errorf("f16[%d] = %v want %v", i, got[i], v)
		}
	}

	// BF16 round-trip.
	bf := make([]byte, 2*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint16(bf[i*2:], float32ToBfloat16(v))
	}
	tn = &Tensor{Name: "t", Info: TensorInfo{Dtype: BF16, Shape: []int64{int64(len(vals))}}, Data: bf}
	got, err = tn.Float32()
	if err != nil {
		t.Fatalf("Float32 bf16: %v", err)
	}
	for i, v := range vals {
		if got[i] != v {
			t.Errorf("bf16[%d] = %v want %v", i, got[i], v)
		}
	}
}

func TestStreamingWriterIter(t *testing.T) {
	var buf bytes.Buffer
	w := NewWriter(&buf, nil)
	if err := w.Declare("a", U8, []int64{3}); err != nil {
		t.Fatal(err)
	}
	if err := w.Declare("b", U8, []int64{2}); err != nil {
		t.Fatal(err)
	}
	if err := w.Commit(); err != nil {
		t.Fatal(err)
	}
	if err := w.Write("a", []byte{1, 2, 3}); err != nil {
		t.Fatal(err)
	}
	// Out-of-order write must fail.
	if err := w.Write("a", []byte{9}); err == nil {
		t.Fatal("expected out-of-order write error")
	}
	if err := w.Write("b", []byte{4, 5}); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	f, err := NewReader(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		t.Fatal(err)
	}
	seq, iterErr := f.Iter()
	collected := map[string][]byte{}
	for name, tn := range seq {
		collected[name] = tn.Data
	}
	if *iterErr != nil {
		t.Fatalf("iter error: %v", *iterErr)
	}
	if !bytes.Equal(collected["a"], []byte{1, 2, 3}) || !bytes.Equal(collected["b"], []byte{4, 5}) {
		t.Errorf("iter collected = %v", collected)
	}
}

func TestRejectsMalformed(t *testing.T) {
	// Header length larger than file.
	var b [16]byte
	binary.LittleEndian.PutUint64(b[:], 1<<40)
	if _, err := NewReader(bytes.NewReader(b[:]), int64(len(b))); err == nil {
		t.Error("expected error for oversized header length")
	}
	// Too small for prefix.
	if _, err := NewReader(bytes.NewReader([]byte{1, 2}), 2); err == nil {
		t.Error("expected error for truncated file")
	}
}

func TestOpenMmap(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "t.safetensors")
	var buf bytes.Buffer
	tensors := []*Tensor{{Name: "x", Info: TensorInfo{Dtype: F32, Shape: []int64{3}}, Data: f32Bytes(1, 2, 3)}}
	if err := Save(&buf, tensors, nil); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer func() { _ = f.Close() }()
	x, err := f.Tensor("x")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(x.Data, f32Bytes(1, 2, 3)) {
		t.Errorf("mmap read mismatch: %v", x.Data)
	}
}
