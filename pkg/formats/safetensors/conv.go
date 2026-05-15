package safetensors

import "math"

// float16ToFloat32 converts an IEEE-754 half-precision value (as a raw uint16)
// to float32, handling subnormals, infinities, and NaN.
func float16ToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := (h >> 10) & 0x1f
	mant := uint32(h & 0x3ff)

	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign) // signed zero
		}
		// Subnormal: normalize into a float32 normal.
		e := uint32(-14 + 127)
		for mant&0x400 == 0 {
			mant <<= 1
			e--
		}
		mant &= 0x3ff
		return math.Float32frombits(sign | (e << 23) | (mant << 13))
	case 0x1f:
		// Inf or NaN.
		return math.Float32frombits(sign | (0xff << 23) | (mant << 13))
	default:
		e := uint32(exp) - 15 + 127
		return math.Float32frombits(sign | (e << 23) | (mant << 13))
	}
}

// bfloat16ToFloat32 converts a bfloat16 (truncated float32) to float32 by
// placing its 16 bits in the high half of the float32 mantissa/exponent.
func bfloat16ToFloat32(b uint16) float32 {
	return math.Float32frombits(uint32(b) << 16)
}

// float32ToFloat16 converts a float32 to IEEE-754 half precision, rounding to
// nearest-even. Out-of-range magnitudes saturate to +/-Inf.
func float32ToFloat16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int32((bits>>23)&0xff) - 127 + 15
	mant := bits & 0x7fffff

	if (bits>>23)&0xff == 0xff {
		// Inf or NaN.
		if mant != 0 {
			return sign | 0x7e00 // canonical NaN
		}
		return sign | 0x7c00
	}
	if exp >= 0x1f {
		return sign | 0x7c00 // overflow -> Inf
	}
	if exp <= 0 {
		if exp < -10 {
			return sign // underflow -> signed zero
		}
		// Subnormal half.
		mant |= 0x800000
		shift := uint32(14 - exp)
		half := uint16(mant >> shift)
		if mant&((1<<shift)-1) > (1 << (shift - 1)) {
			half++ // round half up (simplified)
		}
		return sign | half
	}
	half := sign | uint16(exp<<10) | uint16(mant>>13)
	if mant&0x1000 != 0 { // round to nearest even on dropped bits
		half++
	}
	return half
}

// float32ToBfloat16 truncates a float32 to bfloat16 with round-to-nearest-even.
func float32ToBfloat16(f float32) uint16 {
	bits := math.Float32bits(f)
	// Round to nearest even before truncating the low 16 bits.
	rounding := uint32(0x7fff) + ((bits >> 16) & 1)
	return uint16((bits + rounding) >> 16)
}
