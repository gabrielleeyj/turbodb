package ioformats

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"

	"github.com/gabrielleeyj/turbodb/internal/engine"
	"github.com/gabrielleeyj/turbodb/pkg/formats/safetensors"
	"github.com/gabrielleeyj/turbodb/pkg/index"
)

// ImportOptions configures an import into the engine.
type ImportOptions struct {
	Collection  string
	BitWidth    int    // 1..8; defaults to 4
	IDPrefix    string // vector ids are IDPrefix + row index
	RotatorSeed uint64
	// Progress, if non-nil, is called with the number of vectors inserted so
	// far at periodic intervals.
	Progress func(done, total int)
}

// Importer creates collections and inserts vectors into an engine.
type Importer interface {
	CreateCollection(ctx context.Context, cfg engine.CollectionConfig) error
	Insert(ctx context.Context, collection string, entry index.VectorEntry) error
}

const defaultBitWidth = 4

// ImportMatrix creates a collection sized to the matrix and inserts every row.
// It is the engine side of `turbodb-ctl import`.
func ImportMatrix(ctx context.Context, eng Importer, m Matrix, opts ImportOptions) (int, error) {
	if opts.Collection == "" {
		return 0, fmt.Errorf("ioformats: import requires a collection name")
	}
	if m.Rows == 0 {
		return 0, fmt.Errorf("ioformats: nothing to import (0 rows)")
	}
	bitWidth := opts.BitWidth
	if bitWidth == 0 {
		bitWidth = defaultBitWidth
	}
	cfg := engine.CollectionConfig{
		Name:        opts.Collection,
		Dim:         m.Dim,
		BitWidth:    bitWidth,
		Metric:      engine.MetricInnerProduct,
		Variant:     engine.VariantMSE,
		RotatorSeed: opts.RotatorSeed,
	}
	if err := cfg.Validate(); err != nil {
		return 0, err
	}
	if err := eng.CreateCollection(ctx, cfg); err != nil {
		return 0, fmt.Errorf("ioformats: create collection: %w", err)
	}

	const progressEvery = 1000
	for i := 0; i < m.Rows; i++ {
		entry := index.VectorEntry{
			ID:     fmt.Sprintf("%s%d", opts.IDPrefix, i),
			Values: m.Row(i),
		}
		if err := eng.Insert(ctx, opts.Collection, entry); err != nil {
			return i, fmt.Errorf("ioformats: insert row %d: %w", i, err)
		}
		if opts.Progress != nil && (i+1)%progressEvery == 0 {
			opts.Progress(i+1, m.Rows)
		}
	}
	if opts.Progress != nil {
		opts.Progress(m.Rows, m.Rows)
	}
	return m.Rows, nil
}

// ExportSafeTensors writes the collection's vectors to a SafeTensors stream as a
// single F32 tensor named "vectors" of shape [N, dim], with vector ids recorded
// in the __metadata__ map under "ids" (newline-joined) and the quantization
// parameters under the standard TurboQuant keys.
func ExportSafeTensors(w io.Writer, cfg engine.CollectionConfig, entries []index.VectorEntry) error {
	n := len(entries)
	if n == 0 {
		return fmt.Errorf("ioformats: nothing to export")
	}
	dim := cfg.Dim

	meta := safetensors.QuantMeta{
		RotatorSeed: cfg.RotatorSeed,
		RotatorType: "hadamard",
		CodebookID:  fmt.Sprintf("d%d_b%d_lloyd_max_v1", dim, cfg.BitWidth),
		BitWidth:    cfg.BitWidth,
		Variant:     string(cfg.Variant),
	}.ToMap()

	ids := make([]byte, 0, n*8)
	for i, e := range entries {
		if len(e.Values) != dim {
			return fmt.Errorf("ioformats: row %d has dim %d, expected %d", i, len(e.Values), dim)
		}
		ids = append(ids, []byte(e.ID)...)
		ids = append(ids, '\n')
	}
	meta["ids"] = string(ids)

	sw := safetensors.NewWriter(w, meta)
	if err := sw.Declare("vectors", safetensors.F32, []int64{int64(n), int64(dim)}); err != nil {
		return err
	}
	if err := sw.Commit(); err != nil {
		return err
	}

	buf := make([]byte, n*dim*4)
	for i, e := range entries {
		for j, v := range e.Values {
			binary.LittleEndian.PutUint32(buf[(i*dim+j)*4:], math.Float32bits(v))
		}
	}
	if err := sw.Write("vectors", buf); err != nil {
		return err
	}
	return sw.Close()
}
