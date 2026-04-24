// Package cuda provides Go bindings to the TurboQuant CUDA kernel layer.
//
// Build with -tags cuda to enable GPU acceleration.
// Without the tag, all operations fall back to the CPU reference implementation.
package cuda

import "errors"

// Status codes matching tq_status_t in turboquant.h.
const (
	StatusOK         = 0
	StatusCUDA       = 1
	StatusOOM        = 2
	StatusInvalidArg = 3
	StatusDimNotPow2 = 4
	StatusUnsupported = 5
	StatusNotInit    = 6
	StatusInternal   = 7
)

var (
	ErrCUDA       = errors.New("cuda: CUDA runtime error")
	ErrOOM        = errors.New("cuda: out of memory")
	ErrInvalidArg = errors.New("cuda: invalid argument")
	ErrDimNotPow2 = errors.New("cuda: dimension must be power of 2")
	ErrUnsupported = errors.New("cuda: unsupported configuration")
	ErrNotInit    = errors.New("cuda: context not initialized")
	ErrInternal   = errors.New("cuda: internal error")
	ErrNoCUDA     = errors.New("cuda: not compiled with CUDA support (use -tags cuda)")
)

// statusToError converts a C tq_status_t to a Go error.
func statusToError(status int) error {
	switch status {
	case StatusOK:
		return nil
	case StatusCUDA:
		return ErrCUDA
	case StatusOOM:
		return ErrOOM
	case StatusInvalidArg:
		return ErrInvalidArg
	case StatusDimNotPow2:
		return ErrDimNotPow2
	case StatusUnsupported:
		return ErrUnsupported
	case StatusNotInit:
		return ErrNotInit
	case StatusInternal:
		return ErrInternal
	default:
		return ErrInternal
	}
}
