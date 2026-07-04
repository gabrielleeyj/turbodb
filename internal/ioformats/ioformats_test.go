package ioformats

import (
	"bytes"
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/gabrielleeyj/turbodb/internal/engine"
	"github.com/gabrielleeyj/turbodb/pkg/formats/safetensors"
	"github.com/gabrielleeyj/turbodb/pkg/search"
)

// writeSafeTensorsMatrix writes an [N, dim] F32 tensor to path.
func writeSafeTensorsMatrix(t *testing.T, path string, rows, dim int) {
	t.Helper()
	buf := make([]byte, rows*dim*4)
	for i := 0; i < rows*dim; i++ {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(float32(i%7)))
	}
	var out bytes.Buffer
	tensor := &safetensors.Tensor{
		Name: "embeddings",
		Info: safetensors.TensorInfo{Dtype: safetensors.F32, Shape: []int64{int64(rows), int64(dim)}},
		Data: buf,
	}
	if err := safetensors.Save(&out, []*safetensors.Tensor{tensor}, nil); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, out.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestReadMatrixSafeTensors(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "e.safetensors")
	writeSafeTensorsMatrix(t, path, 5, 4)

	m, err := ReadMatrix(FormatSafeTensors, path, "")
	if err != nil {
		t.Fatalf("ReadMatrix: %v", err)
	}
	if m.Rows != 5 || m.Dim != 4 {
		t.Fatalf("matrix shape = %dx%d", m.Rows, m.Dim)
	}
	if got := m.Row(1)[0]; got != float32(4%7) {
		t.Errorf("row1[0] = %v", got)
	}
}

func TestImportExportRoundTrip(t *testing.T) {
	dir := t.TempDir()
	stPath := filepath.Join(dir, "in.safetensors")
	const rows, dim = 64, 8
	writeSafeTensorsMatrix(t, stPath, rows, dim)

	m, err := ReadMatrix(FormatSafeTensors, stPath, "")
	if err != nil {
		t.Fatal(err)
	}

	eng, err := engine.New(engine.Config{DataDir: filepath.Join(dir, "data")})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = eng.Close() }()

	ctx := context.Background()
	n, err := ImportMatrix(ctx, eng, m, ImportOptions{Collection: "docs", BitWidth: 4})
	if err != nil {
		t.Fatalf("ImportMatrix: %v", err)
	}
	if n != rows {
		t.Fatalf("imported %d, want %d", n, rows)
	}
	if err := eng.Flush(ctx, "docs"); err != nil {
		t.Fatal(err)
	}

	// Search should find a self-match for an imported vector.
	results, _, err := eng.Search(ctx, "docs", m.Row(0), search.Options{TopK: 1})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("got %d results", len(results))
	}

	// Export to SafeTensors and verify the container round-trips.
	cfg, entries, err := eng.ExportCollection("docs")
	if err != nil {
		t.Fatalf("ExportCollection: %v", err)
	}
	if len(entries) != rows {
		t.Fatalf("exported %d entries, want %d", len(entries), rows)
	}
	var exported bytes.Buffer
	if err := ExportSafeTensors(&exported, cfg, entries); err != nil {
		t.Fatalf("ExportSafeTensors: %v", err)
	}
	f, err := safetensors.NewReader(bytes.NewReader(exported.Bytes()), int64(exported.Len()))
	if err != nil {
		t.Fatalf("re-read export: %v", err)
	}
	info, ok := f.Info("vectors")
	if !ok {
		t.Fatal("exported file missing 'vectors' tensor")
	}
	if info.Shape[0] != rows || info.Shape[1] != dim {
		t.Errorf("exported shape = %v", info.Shape)
	}
	meta, err := safetensors.ParseQuantMeta(f.Metadata())
	if err != nil {
		t.Fatal(err)
	}
	if meta.BitWidth != 4 {
		t.Errorf("exported bit width = %d", meta.BitWidth)
	}
}

func TestConvertAndInspect(t *testing.T) {
	dir := t.TempDir()
	stPath := filepath.Join(dir, "c.safetensors")
	writeSafeTensorsMatrix(t, stPath, 3, 4)

	// Inspect SafeTensors.
	tensors, _, err := Inspect(FormatSafeTensors, stPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(tensors) != 1 || tensors[0].Name != "embeddings" {
		t.Fatalf("inspect tensors = %+v", tensors)
	}

	// Convert to GGUF, then inspect the result.
	ggufPath := filepath.Join(dir, "c.gguf")
	out, err := os.Create(ggufPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := ConvertSafeTensorsToGGUF(stPath, out); err != nil {
		_ = out.Close()
		t.Fatalf("convert: %v", err)
	}
	_ = out.Close()

	gtensors, _, err := Inspect(FormatGGUF, ggufPath)
	if err != nil {
		t.Fatalf("inspect gguf: %v", err)
	}
	if len(gtensors) != 1 {
		t.Fatalf("gguf tensors = %+v", gtensors)
	}

	// Read it back as a matrix (dims reversed by ggml convention).
	m, err := ReadMatrix(FormatGGUF, ggufPath, gtensors[0].Name)
	if err != nil {
		t.Fatalf("read gguf matrix: %v", err)
	}
	if m.Dim != 4 || m.Rows != 3 {
		t.Errorf("gguf matrix shape = %dx%d", m.Rows, m.Dim)
	}
}

func TestReadMatrixErrors(t *testing.T) {
	if _, err := ReadMatrix("bogus", "x", ""); err == nil {
		t.Error("expected unsupported-format error")
	}
	if _, err := ReadMatrix(FormatGGUF, "x", ""); err == nil {
		t.Error("expected gguf-requires-tensor error")
	}
	if _, err := ReadMatrix(FormatSafeTensors, "/nonexistent.st", ""); err == nil {
		t.Error("expected open error")
	}
	if _, _, err := Inspect("bogus", "x"); err == nil {
		t.Error("expected inspect unsupported-format error")
	}
}

func TestShape2D(t *testing.T) {
	if r, d, _ := shape2D([]int64{7}); r != 1 || d != 7 {
		t.Errorf("1D shape => %d,%d", r, d)
	}
	if _, _, err := shape2D([]int64{1, 2, 3}); err == nil {
		t.Error("expected 3D shape error")
	}
}

func TestExportEmptyErrors(t *testing.T) {
	var buf bytes.Buffer
	if err := ExportSafeTensors(&buf, engine.CollectionConfig{Dim: 4}, nil); err == nil {
		t.Error("expected empty-export error")
	}
}

func TestImportValidation(t *testing.T) {
	eng, err := engine.New(engine.Config{DataDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = eng.Close() }()
	ctx := context.Background()
	if _, err := ImportMatrix(ctx, eng, Matrix{}, ImportOptions{Collection: "x"}); err == nil {
		t.Error("expected empty-matrix error")
	}
	if _, err := ImportMatrix(ctx, eng, Matrix{Rows: 1, Dim: 1, Values: []float32{1}}, ImportOptions{}); err == nil {
		t.Error("expected missing-collection error")
	}
}
