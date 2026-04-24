package cuda

import (
	"errors"
	"testing"
)

func TestAvailable(t *testing.T) {
	// On a non-CUDA build, Available should be false.
	if Available() {
		t.Skip("CUDA is available — stub tests not applicable")
	}
}

func TestNewContextWithoutCUDA(t *testing.T) {
	if Available() {
		t.Skip("CUDA is available")
	}

	ctx, err := NewContext(0)
	if ctx != nil {
		t.Error("expected nil context without CUDA")
	}
	if !errors.Is(err, ErrNoCUDA) {
		t.Errorf("expected ErrNoCUDA, got %v", err)
	}
}

func TestStatusToError(t *testing.T) {
	tests := []struct {
		status int
		want   error
	}{
		{StatusOK, nil},
		{StatusCUDA, ErrCUDA},
		{StatusOOM, ErrOOM},
		{StatusInvalidArg, ErrInvalidArg},
		{StatusDimNotPow2, ErrDimNotPow2},
		{StatusUnsupported, ErrUnsupported},
		{StatusNotInit, ErrNotInit},
		{StatusInternal, ErrInternal},
		{99, ErrInternal}, // unknown maps to internal
	}

	for _, tt := range tests {
		got := statusToError(tt.status)
		if !errors.Is(got, tt.want) {
			t.Errorf("statusToError(%d) = %v, want %v", tt.status, got, tt.want)
		}
	}
}

func TestPoolWithoutCUDA(t *testing.T) {
	if Available() {
		t.Skip("CUDA is available")
	}

	pool := NewPool(0, 4)
	defer pool.Close()

	if pool.Size() != 0 {
		t.Errorf("expected empty pool, got size %d", pool.Size())
	}

	_, err := pool.Get()
	if !errors.Is(err, ErrNoCUDA) {
		t.Errorf("expected ErrNoCUDA from pool.Get(), got %v", err)
	}
}
