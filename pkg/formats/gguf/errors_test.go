package gguf

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func TestWriterErrorPaths(t *testing.T) {
	w := NewWriter(&bytes.Buffer{})
	if err := w.Declare("a", GGMLTypeF32, []uint64{1}, 4); err != nil {
		t.Fatal(err)
	}
	if err := w.Declare("a", GGMLTypeF32, []uint64{1}, 4); err == nil {
		t.Error("expected duplicate declare error")
	}
	if err := w.Declare("neg", GGMLTypeF32, []uint64{1}, -1); err == nil {
		t.Error("expected negative size error")
	}
	if err := w.WriteData("a", []byte{0, 0, 0, 0}); err == nil {
		t.Error("expected write-before-commit error")
	}
	if err := w.Close(); err == nil {
		t.Error("expected close-before-commit error")
	}
	if err := w.Commit(); err != nil {
		t.Fatal(err)
	}
	if err := w.Declare("late", GGMLTypeF32, []uint64{1}, 4); err == nil {
		t.Error("expected declare-after-commit error")
	}
	if err := w.Commit(); err == nil {
		t.Error("expected double-commit error")
	}
	if err := w.WriteData("wrong", []byte{0, 0, 0, 0}); err == nil {
		t.Error("expected out-of-order error")
	}
	if err := w.WriteData("a", []byte{0, 0}); err == nil {
		t.Error("expected size-mismatch error")
	}
}

func TestOpenAndMetadataTypes(t *testing.T) {
	var buf bytes.Buffer
	w := NewWriter(&buf)
	w.SetMetadata("u8", Value{Type: mvUint8, Num: 7})
	w.SetMetadata("i64", Value{Type: mvInt64, Num: 123})
	w.SetMetadata("f32", Value{Type: mvFloat32, F64: 1.5})
	w.SetMetadata("f64", Value{Type: mvFloat64, F64: 2.5})
	w.SetMetadata("flag", Value{Type: mvBool, Num: 1})
	w.SetMetadata("name", StringValue("hi"))
	// Two tensors to exercise inter-tensor alignment padding.
	if err := w.Declare("a", GGMLTypeF32, []uint64{3}, 12); err != nil {
		t.Fatal(err)
	}
	if err := w.Declare("b", GGMLTypeF32, []uint64{1}, 4); err != nil {
		t.Fatal(err)
	}
	if err := w.Commit(); err != nil {
		t.Fatal(err)
	}
	_ = w.WriteData("a", make([]byte, 12))
	_ = w.WriteData("b", make([]byte, 4))
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "m.gguf")
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer f.Close()

	if f.Alignment() != DefaultAlignment {
		t.Errorf("alignment = %d", f.Alignment())
	}
	if got := len(f.Names()); got != 2 {
		t.Errorf("names = %d", got)
	}
	if keys := f.MetadataKeys(); len(keys) < 6 {
		t.Errorf("metadata keys = %v", keys)
	}
	u8, _ := f.Metadata("u8")
	if n, _ := u8.AsUint64(); n != 7 {
		t.Errorf("u8 = %d", n)
	}
	f32, _ := f.Metadata("f32")
	if v, _ := f32.AsFloat64(); v != 1.5 {
		t.Errorf("f32 = %v", v)
	}
	// b is aligned to 32 after a's 12 bytes.
	bi, _ := f.Info("b")
	if bi.Offset != 32 {
		t.Errorf("b offset = %d, want 32 (aligned)", bi.Offset)
	}
}

func TestValueAccessorErrors(t *testing.T) {
	s := StringValue("x")
	if _, err := s.AsUint64(); err == nil {
		t.Error("string AsUint64 should error")
	}
	if _, err := s.AsFloat64(); err == nil {
		t.Error("string AsFloat64 should error")
	}
	n := Uint64Value(1)
	if _, err := n.AsString(); err == nil {
		t.Error("int AsString should error")
	}
}

func TestUnsupportedDequant(t *testing.T) {
	if _, err := dequantize(GGMLTypeQ6_K, make([]byte, 64), 32); err == nil {
		t.Error("expected unimplemented dequant error")
	}
}

func TestHalfFloatSpecialValues(t *testing.T) {
	for _, h := range []uint16{0x0000, 0x8000, 0x7c00, 0xfc00, 0x0001, 0x03ff, 0x7e00} {
		_ = f16ToF32(h)
	}
	for _, f := range []float32{0, -0, 1e-9, 70000, -70000, 1e30, 5.96e-8} {
		_ = float32ToHalf(f)
	}
}

func TestMoreMetadataIntTypes(t *testing.T) {
	var buf bytes.Buffer
	w := NewWriter(&buf)
	w.SetMetadata("u16", Value{Type: mvUint16, Num: 300})
	w.SetMetadata("i8", Value{Type: mvInt8, Num: 5})
	w.SetMetadata("i16", Value{Type: mvInt16, Num: 9})
	w.SetMetadata("i32", Value{Type: mvInt32, Num: 11})
	_ = w.Declare("t", GGMLTypeF32, []uint64{1}, 4)
	_ = w.Commit()
	_ = w.WriteData("t", []byte{0, 0, 0, 0})
	_ = w.Close()
	f, err := NewReader(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		t.Fatal(err)
	}
	u16, _ := f.Metadata("u16")
	if n, _ := u16.AsUint64(); n != 300 {
		t.Errorf("u16 = %d", n)
	}
}

func TestWriteUnknownValueTypeErrors(t *testing.T) {
	var buf bytes.Buffer
	if err := writeValue(&buf, Value{Type: metadataValueType(99)}); err == nil {
		t.Error("expected unknown value type write error")
	}
}

func TestQ4_1DequantAndShortData(t *testing.T) {
	block := make([]byte, 20)
	binary.LittleEndian.PutUint16(block[0:], f32ToF16(1.0)) // d
	binary.LittleEndian.PutUint16(block[2:], f32ToF16(2.0)) // m
	for i := 4; i < 20; i++ {
		block[i] = 0x00 // nibbles 0 -> value = m = 2.0
	}
	got, err := dequantize(GGMLTypeQ4_1, block, 32)
	if err != nil {
		t.Fatal(err)
	}
	if got[0] != 2.0 {
		t.Errorf("q4_1[0] = %v want 2.0", got[0])
	}
	// Short data must error for each block format.
	if _, err := dequantize(GGMLTypeQ8_0, make([]byte, 4), 32); err == nil {
		t.Error("expected short Q8_0 error")
	}
	if _, err := dequantize(GGMLTypeF32, make([]byte, 4), 32); err == nil {
		t.Error("expected short F32 error")
	}
}

func TestMiscHelpers(t *testing.T) {
	w := NewWriter(&bytes.Buffer{})
	w.SetAlignment(64)
	_ = w.Declare("a", GGMLTypeF32, []uint64{1}, 4)
	if int64(w.tensors["a"].Offset) != 0 {
		t.Errorf("first offset should be 0")
	}
	if keys := SortedMetadataKeys([]string{"b", "a"}); keys[0] != "a" {
		t.Errorf("SortedMetadataKeys not sorted: %v", keys)
	}
	for _, ty := range []GGMLType{GGMLTypeF32, GGMLTypeQ4_K, GGMLTypeTurboQuantProd, GGMLType(999)} {
		if ty.String() == "" {
			t.Errorf("String() empty for %d", ty)
		}
	}
	arr := Value{Type: mvArray}
	if arr.kindName() != "array" {
		t.Errorf("kindName array")
	}
}

func TestRawNotFound(t *testing.T) {
	var buf bytes.Buffer
	w := NewWriter(&buf)
	_ = w.Declare("a", GGMLTypeF32, []uint64{1}, 4)
	_ = w.Commit()
	_ = w.WriteData("a", []byte{0, 0, 0, 0})
	_ = w.Close()
	f, err := NewReader(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.Raw("missing"); err == nil {
		t.Error("expected not-found error")
	}
	if _, err := f.Float32("missing"); err == nil {
		t.Error("expected not-found error")
	}
}
