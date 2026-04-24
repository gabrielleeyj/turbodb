//go:build !cuda

package cuda

import "fmt"

// NewContext returns an error when built without the cuda tag.
func NewContext(deviceID int) (Context, error) {
	return nil, ErrNoCUDA
}

// Available reports whether CUDA support is compiled in.
func Available() bool {
	return false
}

// stubCodebook is a placeholder that satisfies the Codebook interface.
type stubCodebook struct{}

func (stubCodebook) Close()      {}
func (stubCodebook) BitWidth() int { return 0 }
func (stubCodebook) Size() int   { return 0 }

// stubRotator is a placeholder that satisfies the Rotator interface.
type stubRotator struct{}

func (stubRotator) Close()     {}
func (stubRotator) Dim() int   { return 0 }
func (stubRotator) OutDim() int { return 0 }

// stubContext returns ErrNoCUDA for all operations.
type stubContext struct{}

func (stubContext) Close()           {}
func (stubContext) Sync() error      { return ErrNoCUDA }
func (stubContext) LastError() string { return ErrNoCUDA.Error() }

func (stubContext) DeviceInfo() (DeviceInfo, error) {
	return DeviceInfo{}, ErrNoCUDA
}

func (stubContext) CreateCodebook(_ []float32, _ int) (Codebook, error) {
	return nil, ErrNoCUDA
}

func (stubContext) CreateRotator(_ int, _ uint64) (Rotator, error) {
	return nil, ErrNoCUDA
}

func (stubContext) FWHTBatch(_ Rotator, _ []float32, _ int) ([]float32, error) {
	return nil, ErrNoCUDA
}

func (stubContext) FWHTInverseBatch(_ Rotator, _ []float32, _ int) ([]float32, error) {
	return nil, ErrNoCUDA
}

func (stubContext) QuantizeMSEBatch(_ []float32, _, _ int, _ Rotator, _ Codebook) ([]byte, []float32, error) {
	return nil, nil, ErrNoCUDA
}

func (stubContext) DequantizeMSEBatch(_ []byte, _ []float32, _, _ int, _ Rotator, _ Codebook) ([]float32, error) {
	return nil, ErrNoCUDA
}

func (stubContext) QJLSketchBatch(_ []float32, _, _, _ int, _ uint64) ([]byte, []float32, error) {
	return nil, nil, ErrNoCUDA
}

func (stubContext) SearchBruteForce(_ SearchParams) ([][]SearchResult, error) {
	return nil, ErrNoCUDA
}

func init() {
	// Verify stub satisfies interface at compile time.
	var _ Context = stubContext{}
	var _ Codebook = stubCodebook{}
	var _ Rotator = stubRotator{}
	_ = fmt.Sprintf // suppress unused import
}
