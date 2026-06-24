package gguf

import (
	"encoding/binary"
	"fmt"
	"math"
)

func f32FromBits(b uint32) float32 { return math.Float32frombits(b) }
func f64FromBits(b uint64) float64 { return math.Float64frombits(b) }
func float32bits(f float32) uint32 { return math.Float32bits(f) }
func float64bits(f float64) uint64 { return math.Float64bits(f) }
func f32ToF16(f float32) uint16    { return float32ToHalf(f) }

// f16ToF32 converts an IEEE-754 half to float32.
func f16ToF32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := (h >> 10) & 0x1f
	mant := uint32(h & 0x3ff)
	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		e := uint32(-14 + 127)
		for mant&0x400 == 0 {
			mant <<= 1
			e--
		}
		mant &= 0x3ff
		return math.Float32frombits(sign | (e << 23) | (mant << 13))
	case 0x1f:
		return math.Float32frombits(sign | (0xff << 23) | (mant << 13))
	default:
		e := uint32(exp) - 15 + 127
		return math.Float32frombits(sign | (e << 23) | (mant << 13))
	}
}

// float32ToHalf converts a float32 to IEEE-754 half precision with round to
// nearest even; out-of-range magnitudes saturate to +/-Inf.
func float32ToHalf(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int32((bits>>23)&0xff) - 127 + 15
	mant := bits & 0x7fffff
	if (bits>>23)&0xff == 0xff {
		if mant != 0 {
			return sign | 0x7e00
		}
		return sign | 0x7c00
	}
	if exp >= 0x1f {
		return sign | 0x7c00
	}
	if exp <= 0 {
		if exp < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint32(14 - exp)
		half := uint16(mant >> shift)
		if mant&((1<<shift)-1) > (1 << (shift - 1)) {
			half++
		}
		return sign | half
	}
	half := sign | uint16(exp<<10) | uint16(mant>>13)
	if mant&0x1000 != 0 {
		half++
	}
	return half
}

const qkBlock = 32 // elements per Q4_0/Q4_1/Q8_0 block

// dequantize expands raw block-encoded bytes into n float32 values.
func dequantize(t GGMLType, raw []byte, n int) ([]float32, error) {
	out := make([]float32, n)
	switch t {
	case GGMLTypeF32:
		if len(raw) < n*4 {
			return nil, errShort(t, len(raw), n*4)
		}
		for i := 0; i < n; i++ {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	case GGMLTypeF16:
		if len(raw) < n*2 {
			return nil, errShort(t, len(raw), n*2)
		}
		for i := 0; i < n; i++ {
			out[i] = f16ToF32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
	case GGMLTypeQ8_0:
		return dequantQ8_0(raw, n)
	case GGMLTypeQ4_0:
		return dequantQ4_0(raw, n)
	case GGMLTypeQ4_1:
		return dequantQ4_1(raw, n)
	default:
		return nil, fmt.Errorf("gguf: dequantize of type %s not implemented", t)
	}
	return out, nil
}

func errShort(t GGMLType, got, want int) error {
	return fmt.Errorf("gguf: %s data too short: %d bytes, need %d", t, got, want)
}

// dequantQ8_0: each 34-byte block = f16 scale d + 32 int8 quants; x = d*q.
func dequantQ8_0(raw []byte, n int) ([]float32, error) {
	const blkBytes = 34
	out := make([]float32, n)
	for b := 0; b*qkBlock < n; b++ {
		base := b * blkBytes
		if base+blkBytes > len(raw) {
			return nil, errShort(GGMLTypeQ8_0, len(raw), (b+1)*blkBytes)
		}
		d := f16ToF32(binary.LittleEndian.Uint16(raw[base:]))
		for j := 0; j < qkBlock; j++ {
			out[b*qkBlock+j] = d * float32(int8(raw[base+2+j]))
		}
	}
	return out, nil
}

// dequantQ4_0: each 18-byte block = f16 scale d + 16 packed nibbles;
// x = d * (nibble - 8).
func dequantQ4_0(raw []byte, n int) ([]float32, error) {
	const blkBytes = 18
	out := make([]float32, n)
	for b := 0; b*qkBlock < n; b++ {
		base := b * blkBytes
		if base+blkBytes > len(raw) {
			return nil, errShort(GGMLTypeQ4_0, len(raw), (b+1)*blkBytes)
		}
		d := f16ToF32(binary.LittleEndian.Uint16(raw[base:]))
		qs := raw[base+2 : base+18]
		for j := 0; j < qkBlock/2; j++ {
			lo := int(qs[j]&0x0f) - 8
			hi := int(qs[j]>>4) - 8
			out[b*qkBlock+j] = d * float32(lo)
			out[b*qkBlock+j+qkBlock/2] = d * float32(hi)
		}
	}
	return out, nil
}

// dequantQ4_1: each 20-byte block = f16 scale d + f16 min m + 16 packed
// nibbles; x = d*nibble + m.
func dequantQ4_1(raw []byte, n int) ([]float32, error) {
	const blkBytes = 20
	out := make([]float32, n)
	for b := 0; b*qkBlock < n; b++ {
		base := b * blkBytes
		if base+blkBytes > len(raw) {
			return nil, errShort(GGMLTypeQ4_1, len(raw), (b+1)*blkBytes)
		}
		d := f16ToF32(binary.LittleEndian.Uint16(raw[base:]))
		m := f16ToF32(binary.LittleEndian.Uint16(raw[base+2:]))
		qs := raw[base+4 : base+20]
		for j := 0; j < qkBlock/2; j++ {
			lo := int(qs[j] & 0x0f)
			hi := int(qs[j] >> 4)
			out[b*qkBlock+j] = d*float32(lo) + m
			out[b*qkBlock+j+qkBlock/2] = d*float32(hi) + m
		}
	}
	return out, nil
}
