//go:build cuda

package cuda

/*
#cgo LDFLAGS: -L${SRCDIR}/../../cuda/lib -lturboquant_cuda -lcudart -lcublas
#cgo CFLAGS: -I${SRCDIR}/../../cuda/include

#include "turboquant.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"
)

// Available reports whether CUDA support is compiled in.
func Available() bool {
	return true
}

// NewContext creates a CUDA context on the given device.
func NewContext(deviceID int) (Context, error) {
	var handle C.tq_context_t
	status := C.tq_init(C.int(deviceID), &handle)
	if err := statusToError(int(status)); err != nil {
		return nil, fmt.Errorf("cuda: init device %d: %w", deviceID, err)
	}
	ctx := &cgoContext{handle: handle}
	runtime.SetFinalizer(ctx, func(c *cgoContext) { c.Close() })
	return ctx, nil
}

// cgoContext wraps a tq_context_t.
type cgoContext struct {
	handle C.tq_context_t
}

func (c *cgoContext) Close() {
	if c.handle != nil {
		C.tq_destroy(c.handle)
		c.handle = nil
	}
}

func (c *cgoContext) Sync() error {
	return statusToError(int(C.tq_sync(c.handle)))
}

func (c *cgoContext) LastError() string {
	return C.GoString(C.tq_last_error(c.handle))
}

func (c *cgoContext) DeviceInfo() (DeviceInfo, error) {
	var free, total C.size_t
	status := C.tq_device_memory_info(c.handle, &free, &total)
	if err := statusToError(int(status)); err != nil {
		return DeviceInfo{}, err
	}
	var cc C.int
	status = C.tq_device_compute_capability(c.handle, &cc)
	if err := statusToError(int(status)); err != nil {
		return DeviceInfo{}, err
	}
	return DeviceInfo{
		FreeBytes:         uint64(free),
		TotalBytes:        uint64(total),
		ComputeCapability: int(cc),
	}, nil
}

// --- Codebook ---

type cgoCodebook struct {
	handle C.tq_codebook_t
}

func (cb *cgoCodebook) Close() {
	if cb.handle != nil {
		C.tq_codebook_destroy(cb.handle)
		cb.handle = nil
	}
}

func (cb *cgoCodebook) BitWidth() int { return int(C.tq_codebook_bit_width(cb.handle)) }
func (cb *cgoCodebook) Size() int     { return int(C.tq_codebook_size(cb.handle)) }

func (c *cgoContext) CreateCodebook(centroids []float32, bitWidth int) (Codebook, error) {
	if len(centroids) == 0 {
		return nil, fmt.Errorf("cuda: empty centroids")
	}
	var handle C.tq_codebook_t
	status := C.tq_codebook_create(
		c.handle,
		(*C.float)(unsafe.Pointer(&centroids[0])),
		C.int(bitWidth),
		&handle,
	)
	if err := statusToError(int(status)); err != nil {
		return nil, fmt.Errorf("cuda: create codebook: %w", err)
	}
	cb := &cgoCodebook{handle: handle}
	runtime.SetFinalizer(cb, func(cb *cgoCodebook) { cb.Close() })
	return cb, nil
}

// --- Rotator ---

type cgoRotator struct {
	handle C.tq_rotator_t
}

func (r *cgoRotator) Close() {
	if r.handle != nil {
		C.tq_rotator_destroy(r.handle)
		r.handle = nil
	}
}

func (r *cgoRotator) Dim() int    { return int(C.tq_rotator_dim(r.handle)) }
func (r *cgoRotator) OutDim() int { return int(C.tq_rotator_out_dim(r.handle)) }

func (c *cgoContext) CreateRotator(dim int, seed uint64) (Rotator, error) {
	var handle C.tq_rotator_t
	status := C.tq_rotator_create(
		c.handle,
		C.int(dim),
		C.uint64_t(seed),
		&handle,
	)
	if err := statusToError(int(status)); err != nil {
		return nil, fmt.Errorf("cuda: create rotator: %w", err)
	}
	rot := &cgoRotator{handle: handle}
	runtime.SetFinalizer(rot, func(r *cgoRotator) { r.Close() })
	return rot, nil
}

// --- FWHT ---

func (c *cgoContext) FWHTBatch(rot Rotator, input []float32, n int) ([]float32, error) {
	r := rot.(*cgoRotator)
	outDim := r.OutDim()
	output := make([]float32, n*outDim)

	var inputPtr, outputPtr *C.float
	if len(input) > 0 {
		inputPtr = (*C.float)(unsafe.Pointer(&input[0]))
	}
	if len(output) > 0 {
		outputPtr = (*C.float)(unsafe.Pointer(&output[0]))
	}

	// Upload input to device.
	var inputD, outputD unsafe.Pointer
	size := C.size_t(len(input) * 4)

	status := C.tq_device_malloc(c.handle, size, &inputD)
	if err := statusToError(int(status)); err != nil {
		return nil, fmt.Errorf("cuda: fwht alloc input: %w", err)
	}
	defer C.tq_device_free(c.handle, inputD)

	status = C.tq_device_malloc(c.handle, C.size_t(len(output)*4), &outputD)
	if err := statusToError(int(status)); err != nil {
		return nil, fmt.Errorf("cuda: fwht alloc output: %w", err)
	}
	defer C.tq_device_free(c.handle, outputD)

	status = C.tq_memcpy_h2d(c.handle, inputD, unsafe.Pointer(inputPtr), size)
	if err := statusToError(int(status)); err != nil {
		return nil, fmt.Errorf("cuda: fwht upload: %w", err)
	}

	status = C.tq_fwht_batch(c.handle, r.handle,
		(*C.float)(inputD), C.int(n), (*C.float)(outputD))
	if err := statusToError(int(status)); err != nil {
		return nil, fmt.Errorf("cuda: fwht: %w", err)
	}

	status = C.tq_memcpy_d2h(c.handle, unsafe.Pointer(outputPtr), outputD,
		C.size_t(len(output)*4))
	if err := statusToError(int(status)); err != nil {
		return nil, fmt.Errorf("cuda: fwht download: %w", err)
	}

	if err := c.Sync(); err != nil {
		return nil, fmt.Errorf("cuda: fwht sync: %w", err)
	}

	return output, nil
}

func (c *cgoContext) FWHTInverseBatch(rot Rotator, input []float32, n int) ([]float32, error) {
	r := rot.(*cgoRotator)
	outDim := r.OutDim()
	output := make([]float32, n*outDim)

	var inputD, outputD unsafe.Pointer
	inSize := C.size_t(len(input) * 4)
	outSize := C.size_t(len(output) * 4)

	status := C.tq_device_malloc(c.handle, inSize, &inputD)
	if err := statusToError(int(status)); err != nil {
		return nil, err
	}
	defer C.tq_device_free(c.handle, inputD)

	status = C.tq_device_malloc(c.handle, outSize, &outputD)
	if err := statusToError(int(status)); err != nil {
		return nil, err
	}
	defer C.tq_device_free(c.handle, outputD)

	status = C.tq_memcpy_h2d(c.handle, inputD,
		unsafe.Pointer(&input[0]), inSize)
	if err := statusToError(int(status)); err != nil {
		return nil, err
	}

	status = C.tq_fwht_inverse_batch(c.handle, r.handle,
		(*C.float)(inputD), C.int(n), (*C.float)(outputD))
	if err := statusToError(int(status)); err != nil {
		return nil, err
	}

	status = C.tq_memcpy_d2h(c.handle, unsafe.Pointer(&output[0]),
		outputD, outSize)
	if err := statusToError(int(status)); err != nil {
		return nil, err
	}

	if err := c.Sync(); err != nil {
		return nil, err
	}

	return output, nil
}

// --- Quantize / Dequantize ---

func (c *cgoContext) QuantizeMSEBatch(vectors []float32, n, dim int, rot Rotator, cb Codebook) ([]byte, []float32, error) {
	r := rot.(*cgoRotator)
	cbook := cb.(*cgoCodebook)
	outDim := r.OutDim()
	bitWidth := cbook.BitWidth()
	codeBytes := (bitWidth*outDim + 7) / 8

	// Allocate device memory.
	var vecD, codesD, normsD unsafe.Pointer
	vecSize := C.size_t(len(vectors) * 4)
	codesSize := C.size_t(n * codeBytes)
	normsSize := C.size_t(n * 4)

	status := C.tq_device_malloc(c.handle, vecSize, &vecD)
	if err := statusToError(int(status)); err != nil {
		return nil, nil, fmt.Errorf("cuda: quantize alloc vectors: %w", err)
	}
	defer C.tq_device_free(c.handle, vecD)

	status = C.tq_device_malloc(c.handle, codesSize, &codesD)
	if err := statusToError(int(status)); err != nil {
		return nil, nil, fmt.Errorf("cuda: quantize alloc codes: %w", err)
	}
	defer C.tq_device_free(c.handle, codesD)

	status = C.tq_device_malloc(c.handle, normsSize, &normsD)
	if err := statusToError(int(status)); err != nil {
		return nil, nil, fmt.Errorf("cuda: quantize alloc norms: %w", err)
	}
	defer C.tq_device_free(c.handle, normsD)

	// Upload vectors.
	status = C.tq_memcpy_h2d(c.handle, vecD, unsafe.Pointer(&vectors[0]), vecSize)
	if err := statusToError(int(status)); err != nil {
		return nil, nil, err
	}

	// Run quantization.
	status = C.tq_quantize_mse_batch(c.handle,
		(*C.float)(vecD), C.int(n), C.int(dim),
		r.handle, cbook.handle,
		(*C.uint8_t)(codesD), (*C.float)(normsD))
	if err := statusToError(int(status)); err != nil {
		return nil, nil, fmt.Errorf("cuda: quantize: %w", err)
	}

	// Download results.
	codes := make([]byte, n*codeBytes)
	norms := make([]float32, n)

	status = C.tq_memcpy_d2h(c.handle, unsafe.Pointer(&codes[0]), codesD, codesSize)
	if err := statusToError(int(status)); err != nil {
		return nil, nil, err
	}

	status = C.tq_memcpy_d2h(c.handle, unsafe.Pointer(&norms[0]), normsD, normsSize)
	if err := statusToError(int(status)); err != nil {
		return nil, nil, err
	}

	if err := c.Sync(); err != nil {
		return nil, nil, err
	}

	return codes, norms, nil
}

func (c *cgoContext) DequantizeMSEBatch(codes []byte, norms []float32, n, dim int, rot Rotator, cb Codebook) ([]float32, error) {
	r := rot.(*cgoRotator)
	cbook := cb.(*cgoCodebook)

	// Allocate device memory.
	var codesD, normsD, vecD unsafe.Pointer
	codesSize := C.size_t(len(codes))
	normsSize := C.size_t(n * 4)
	vecSize := C.size_t(n * dim * 4)

	status := C.tq_device_malloc(c.handle, codesSize, &codesD)
	if err := statusToError(int(status)); err != nil {
		return nil, err
	}
	defer C.tq_device_free(c.handle, codesD)

	status = C.tq_device_malloc(c.handle, normsSize, &normsD)
	if err := statusToError(int(status)); err != nil {
		return nil, err
	}
	defer C.tq_device_free(c.handle, normsD)

	status = C.tq_device_malloc(c.handle, vecSize, &vecD)
	if err := statusToError(int(status)); err != nil {
		return nil, err
	}
	defer C.tq_device_free(c.handle, vecD)

	// Upload codes and norms.
	status = C.tq_memcpy_h2d(c.handle, codesD, unsafe.Pointer(&codes[0]), codesSize)
	if err := statusToError(int(status)); err != nil {
		return nil, err
	}

	status = C.tq_memcpy_h2d(c.handle, normsD, unsafe.Pointer(&norms[0]), normsSize)
	if err := statusToError(int(status)); err != nil {
		return nil, err
	}

	// Run dequantization.
	status = C.tq_dequantize_mse_batch(c.handle,
		(*C.uint8_t)(codesD), (*C.float)(normsD),
		C.int(n), C.int(dim),
		r.handle, cbook.handle,
		(*C.float)(vecD))
	if err := statusToError(int(status)); err != nil {
		return nil, fmt.Errorf("cuda: dequantize: %w", err)
	}

	// Download results.
	vectors := make([]float32, n*dim)
	status = C.tq_memcpy_d2h(c.handle, unsafe.Pointer(&vectors[0]), vecD, vecSize)
	if err := statusToError(int(status)); err != nil {
		return nil, err
	}

	if err := c.Sync(); err != nil {
		return nil, err
	}

	return vectors, nil
}

// --- QJL ---

func (c *cgoContext) QJLSketchBatch(vectors []float32, n, dim, projDim int, seed uint64) ([]byte, []float32, error) {
	signBytes := (projDim + 7) / 8

	var vecD, signsD, normsD unsafe.Pointer
	vecSize := C.size_t(len(vectors) * 4)
	signsSize := C.size_t(n * signBytes)
	normsSize := C.size_t(n * 4)

	status := C.tq_device_malloc(c.handle, vecSize, &vecD)
	if err := statusToError(int(status)); err != nil {
		return nil, nil, err
	}
	defer C.tq_device_free(c.handle, vecD)

	status = C.tq_device_malloc(c.handle, signsSize, &signsD)
	if err := statusToError(int(status)); err != nil {
		return nil, nil, err
	}
	defer C.tq_device_free(c.handle, signsD)

	status = C.tq_device_malloc(c.handle, normsSize, &normsD)
	if err := statusToError(int(status)); err != nil {
		return nil, nil, err
	}
	defer C.tq_device_free(c.handle, normsD)

	status = C.tq_memcpy_h2d(c.handle, vecD, unsafe.Pointer(&vectors[0]), vecSize)
	if err := statusToError(int(status)); err != nil {
		return nil, nil, err
	}

	status = C.tq_qjl_sketch_batch(c.handle,
		(*C.float)(vecD), C.int(n), C.int(dim),
		C.int(projDim), C.uint64_t(seed),
		(*C.uint8_t)(signsD), (*C.float)(normsD))
	if err := statusToError(int(status)); err != nil {
		return nil, nil, fmt.Errorf("cuda: qjl sketch: %w", err)
	}

	signs := make([]byte, n*signBytes)
	norms := make([]float32, n)

	status = C.tq_memcpy_d2h(c.handle, unsafe.Pointer(&signs[0]), signsD, signsSize)
	if err := statusToError(int(status)); err != nil {
		return nil, nil, err
	}

	status = C.tq_memcpy_d2h(c.handle, unsafe.Pointer(&norms[0]), normsD, normsSize)
	if err := statusToError(int(status)); err != nil {
		return nil, nil, err
	}

	if err := c.Sync(); err != nil {
		return nil, nil, err
	}

	return signs, norms, nil
}

// --- Search ---

func (c *cgoContext) SearchBruteForce(params SearchParams) ([][]SearchResult, error) {
	if params.NQueries <= 0 || params.NDB <= 0 || params.K <= 0 {
		return nil, fmt.Errorf("cuda: search: invalid params")
	}

	cbook := params.CB.(*cgoCodebook)

	// Allocate and upload all buffers.
	type devBuf struct {
		ptr  unsafe.Pointer
		size C.size_t
	}
	var bufs []devBuf

	alloc := func(data unsafe.Pointer, sz int) (unsafe.Pointer, error) {
		var d unsafe.Pointer
		s := C.tq_device_malloc(c.handle, C.size_t(sz), &d)
		if err := statusToError(int(s)); err != nil {
			return nil, err
		}
		bufs = append(bufs, devBuf{d, C.size_t(sz)})
		if data != nil {
			s = C.tq_memcpy_h2d(c.handle, d, data, C.size_t(sz))
			if err := statusToError(int(s)); err != nil {
				return nil, err
			}
		}
		return d, nil
	}

	defer func() {
		for _, b := range bufs {
			C.tq_device_free(c.handle, b.ptr)
		}
	}()

	qCodesD, err := alloc(unsafe.Pointer(&params.QueryCodes[0]), len(params.QueryCodes))
	if err != nil {
		return nil, err
	}

	qNormsD, err := alloc(unsafe.Pointer(&params.QueryNorms[0]), len(params.QueryNorms)*4)
	if err != nil {
		return nil, err
	}

	dbCodesD, err := alloc(unsafe.Pointer(&params.DBCodes[0]), len(params.DBCodes))
	if err != nil {
		return nil, err
	}

	dbNormsD, err := alloc(unsafe.Pointer(&params.DBNorms[0]), len(params.DBNorms)*4)
	if err != nil {
		return nil, err
	}

	// Optional QJL buffers.
	var qSignsD, qResNormsD, dbSignsD, dbResNormsD unsafe.Pointer
	if params.QuerySigns != nil {
		qSignsD, err = alloc(unsafe.Pointer(&params.QuerySigns[0]), len(params.QuerySigns))
		if err != nil {
			return nil, err
		}
	}
	if params.QueryResNorms != nil {
		qResNormsD, err = alloc(unsafe.Pointer(&params.QueryResNorms[0]), len(params.QueryResNorms)*4)
		if err != nil {
			return nil, err
		}
	}
	if params.DBSigns != nil {
		dbSignsD, err = alloc(unsafe.Pointer(&params.DBSigns[0]), len(params.DBSigns))
		if err != nil {
			return nil, err
		}
	}
	if params.DBResNorms != nil {
		dbResNormsD, err = alloc(unsafe.Pointer(&params.DBResNorms[0]), len(params.DBResNorms)*4)
		if err != nil {
			return nil, err
		}
	}

	// Allocate host results buffer.
	totalResults := params.NQueries * params.K
	results := make([]C.tq_search_result_t, totalResults)

	status := C.tq_search_brute_force(c.handle,
		(*C.uint8_t)(qCodesD), (*C.float)(qNormsD),
		(*C.uint8_t)(qSignsD), (*C.float)(qResNormsD),
		C.int(params.NQueries),
		(*C.uint8_t)(dbCodesD), (*C.float)(dbNormsD),
		(*C.uint8_t)(dbSignsD), (*C.float)(dbResNormsD),
		C.int(params.NDB),
		C.int(params.Dim), C.int(params.BitWidth),
		C.int(params.ProjDim),
		cbook.handle,
		C.int(params.K),
		&results[0])
	if err := statusToError(int(status)); err != nil {
		return nil, fmt.Errorf("cuda: search: %w", err)
	}

	// Convert to Go types.
	out := make([][]SearchResult, params.NQueries)
	for q := range params.NQueries {
		out[q] = make([]SearchResult, params.K)
		for k := range params.K {
			r := results[q*params.K+k]
			out[q][k] = SearchResult{
				ID:    int64(r.id),
				Score: float32(r.score),
			}
		}
	}

	return out, nil
}
