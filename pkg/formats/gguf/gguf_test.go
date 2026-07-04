package gguf

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

func TestWriteReadRoundTrip(t *testing.T) {
	// Arrange: two F16 tensors plus a custom TurboQuant tensor (last, so its
	// boundary length equals its exact size with no trailing pad).
	f16data := make([]byte, 64) // 32 F16 elements
	for i := 0; i < 32; i++ {
		binary.LittleEndian.PutUint16(f16data[i*2:], f32ToF16(float32(i)))
	}
	tqBlock, err := EncodeTurboQuantBlock(2.5, 7, makeCodes(), 4)
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	w := NewWriter(&buf)
	w.SetMetadata("general.name", StringValue("turbodb-test"))
	w.SetMetadata("turboquant.bit_width", Uint32Value(4))
	if err := w.Declare("embedding.weight", GGMLTypeF16, []uint64{32}, int64(len(f16data))); err != nil {
		t.Fatal(err)
	}
	if err := w.Declare("codes", GGMLTypeTurboQuantMSE, []uint64{32}, int64(len(tqBlock))); err != nil {
		t.Fatal(err)
	}
	if err := w.Commit(); err != nil {
		t.Fatal(err)
	}
	if err := w.WriteData("embedding.weight", f16data); err != nil {
		t.Fatal(err)
	}
	if err := w.WriteData("codes", tqBlock); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}

	// Act
	f, err := NewReader(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		t.Fatalf("NewReader: %v", err)
	}

	// Assert: metadata
	name, _ := f.Metadata("general.name")
	if s, _ := name.AsString(); s != "turbodb-test" {
		t.Errorf("general.name = %q", s)
	}
	bw, _ := f.Metadata("turboquant.bit_width")
	if n, _ := bw.AsUint64(); n != 4 {
		t.Errorf("bit_width = %d", n)
	}

	// Assert: F16 dequant
	got, err := f.Float32("embedding.weight")
	if err != nil {
		t.Fatalf("Float32: %v", err)
	}
	for i := 0; i < 32; i++ {
		if got[i] != float32(i) {
			t.Errorf("embedding[%d] = %v want %v", i, got[i], float32(i))
		}
	}

	// Assert: custom tensor raw round-trip + block decode
	raw, err := f.Raw("codes")
	if err != nil {
		t.Fatalf("Raw codes: %v", err)
	}
	if !bytes.Equal(raw, tqBlock) {
		t.Fatalf("custom tensor raw mismatch")
	}
	norm, seed, codes, err := DecodeTurboQuantBlock(raw, 4)
	if err != nil {
		t.Fatal(err)
	}
	if seed != 7 || math.Abs(float64(norm-2.5)) > 0.01 {
		t.Errorf("decoded norm=%v seed=%d", norm, seed)
	}
	for i, c := range makeCodes() {
		if codes[i] != c {
			t.Errorf("code[%d] = %d want %d", i, codes[i], c)
		}
	}
}

func makeCodes() []uint8 {
	codes := make([]uint8, TQBlockElems)
	for i := range codes {
		codes[i] = uint8(i % 16) // 4-bit values
	}
	return codes
}

func TestQ8_0Dequant(t *testing.T) {
	// One Q8_0 block: scale 0.5, quants 0..31 -> values 0,0.5,...
	block := make([]byte, 34)
	binary.LittleEndian.PutUint16(block[0:], f32ToF16(0.5))
	for i := 0; i < 32; i++ {
		block[2+i] = byte(int8(i))
	}
	got, err := dequantize(GGMLTypeQ8_0, block, 32)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 32; i++ {
		if want := 0.5 * float32(i); got[i] != want {
			t.Errorf("q8_0[%d] = %v want %v", i, got[i], want)
		}
	}
}

func TestQ4_0Dequant(t *testing.T) {
	block := make([]byte, 18)
	binary.LittleEndian.PutUint16(block[0:], f32ToF16(1.0))
	// nibbles all 8 -> (8-8)=0
	for i := 2; i < 18; i++ {
		block[i] = 0x88
	}
	got, err := dequantize(GGMLTypeQ4_0, block, 32)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 32; i++ {
		if got[i] != 0 {
			t.Errorf("q4_0[%d] = %v want 0", i, got[i])
		}
	}
}

func TestRejectsBadMagic(t *testing.T) {
	bad := make([]byte, 24)
	binary.LittleEndian.PutUint32(bad[0:], 0xdeadbeef)
	if _, err := NewReader(bytes.NewReader(bad), int64(len(bad))); err == nil {
		t.Error("expected bad magic error")
	}
}

func TestRejectsBadVersion(t *testing.T) {
	b := make([]byte, 24)
	binary.LittleEndian.PutUint32(b[0:], Magic)
	binary.LittleEndian.PutUint32(b[4:], 99)
	if _, err := NewReader(bytes.NewReader(b), int64(len(b))); err == nil {
		t.Error("expected bad version error")
	}
}

func TestTurboQuantBlockBitWidths(t *testing.T) {
	for b := 1; b <= 8; b++ {
		codes := make([]uint8, TQBlockElems)
		mask := uint8((1 << b) - 1)
		for i := range codes {
			codes[i] = uint8(i) & mask
		}
		blk, err := EncodeTurboQuantBlock(1.5, uint16(b), codes, b)
		if err != nil {
			t.Fatalf("b=%d encode: %v", b, err)
		}
		if len(blk) != TQBlockBytes(b) {
			t.Errorf("b=%d block bytes = %d want %d", b, len(blk), TQBlockBytes(b))
		}
		_, seed, gotCodes, err := DecodeTurboQuantBlock(blk, b)
		if err != nil {
			t.Fatalf("b=%d decode: %v", b, err)
		}
		if seed != uint16(b) {
			t.Errorf("b=%d seed = %d", b, seed)
		}
		for i := range codes {
			if gotCodes[i] != codes[i] {
				t.Errorf("b=%d code[%d] = %d want %d", b, i, gotCodes[i], codes[i])
			}
		}
	}
}

func TestArrayMetadataRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	w := NewWriter(&buf)
	w.SetMetadata("tokenizer.tokens", Value{
		Type:      mvArray,
		ArrayType: mvString,
		Array:     []Value{StringValue("a"), StringValue("b")},
	})
	if err := w.Declare("t", GGMLTypeF32, []uint64{1}, 4); err != nil {
		t.Fatal(err)
	}
	if err := w.Commit(); err != nil {
		t.Fatal(err)
	}
	if err := w.WriteData("t", []byte{0, 0, 0, 0}); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	f, err := NewReader(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		t.Fatal(err)
	}
	v, ok := f.Metadata("tokenizer.tokens")
	if !ok || v.Type != mvArray || len(v.Array) != 2 {
		t.Fatalf("array metadata = %+v ok=%v", v, ok)
	}
	if s, _ := v.Array[1].AsString(); s != "b" {
		t.Errorf("array[1] = %q", s)
	}
}
