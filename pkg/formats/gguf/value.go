package gguf

import "fmt"

// Value is a typed GGUF metadata value. Exactly one accessor is meaningful per
// instance, selected by Type. Arrays carry their element kind in ArrayType.
type Value struct {
	Type      metadataValueType
	ArrayType metadataValueType // valid when Type == mvArray

	Str    string
	Num    uint64  // raw bits for integer/bool kinds
	F64    float64 // float kinds
	Array  []Value // mvArray
}

// AsString returns the string value, or an error if the value is not a string.
func (v Value) AsString() (string, error) {
	if v.Type != mvString {
		return "", fmt.Errorf("gguf: value is %s, not string", v.kindName())
	}
	return v.Str, nil
}

// AsUint64 coerces any integer/bool value to uint64.
func (v Value) AsUint64() (uint64, error) {
	switch v.Type {
	case mvUint8, mvInt8, mvUint16, mvInt16, mvUint32, mvInt32, mvUint64, mvInt64, mvBool:
		return v.Num, nil
	default:
		return 0, fmt.Errorf("gguf: value is %s, not an integer", v.kindName())
	}
}

// AsFloat64 returns float metadata values.
func (v Value) AsFloat64() (float64, error) {
	if v.Type != mvFloat32 && v.Type != mvFloat64 {
		return 0, fmt.Errorf("gguf: value is %s, not a float", v.kindName())
	}
	return v.F64, nil
}

func (v Value) kindName() string {
	switch v.Type {
	case mvString:
		return "string"
	case mvArray:
		return "array"
	case mvFloat32, mvFloat64:
		return "float"
	case mvBool:
		return "bool"
	default:
		return "int"
	}
}

// StringValue constructs a string-typed metadata Value.
func StringValue(s string) Value { return Value{Type: mvString, Str: s} }

// Uint32Value constructs a uint32-typed metadata Value.
func Uint32Value(n uint32) Value { return Value{Type: mvUint32, Num: uint64(n)} }

// Uint64Value constructs a uint64-typed metadata Value.
func Uint64Value(n uint64) Value { return Value{Type: mvUint64, Num: n} }
