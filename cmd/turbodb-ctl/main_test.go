package main

import (
	"bytes"
	"encoding/binary"
	"math"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"

	apiv1 "github.com/gabrielleeyj/turbodb/api/v1"
	"github.com/gabrielleeyj/turbodb/internal/engine"
	"github.com/gabrielleeyj/turbodb/pkg/formats/safetensors"
	"google.golang.org/grpc"
)

// startTestEngine serves a real engine over gRPC on a random port.
func startTestEngine(t *testing.T) string {
	t.Helper()
	eng, err := engine.New(engine.Config{DataDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = eng.Close() })

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := grpc.NewServer()
	apiv1.RegisterTurboDBEngineServer(srv, engine.NewGRPCServer(eng))
	apiv1.RegisterTurboDBAdminServer(srv, engine.NewAdminServer(eng, "test"))
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.GracefulStop)
	return lis.Addr().String()
}

// execute runs the root command with args and returns combined output.
func execute(t *testing.T, args ...string) (string, error) {
	t.Helper()
	root := newRootCmd()
	var buf bytes.Buffer
	root.SetOut(&buf)
	root.SetErr(&buf)
	root.SetArgs(args)
	err := root.Execute()
	return buf.String(), err
}

func TestVersionCommand(t *testing.T) {
	out, err := execute(t, "version")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "turbodb-ctl") {
		t.Errorf("output: %q", out)
	}
}

func TestCollectionLifecycle(t *testing.T) {
	addr := startTestEngine(t)

	out, err := execute(t, "collection", "create", "--engine", addr,
		"--name", "docs", "--dim", "8", "--bits", "4")
	if err != nil {
		t.Fatalf("create: %v (%s)", err, out)
	}
	if !strings.Contains(out, `collection "docs" created`) {
		t.Errorf("create output: %q", out)
	}

	out, err = execute(t, "collection", "list", "--engine", addr)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "docs") || !strings.Contains(out, "dim=8") {
		t.Errorf("list output: %q", out)
	}

	out, err = execute(t, "collection", "describe", "docs", "--engine", addr)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "dimension:         8") || !strings.Contains(out, "vector_count:      0") {
		t.Errorf("describe output: %q", out)
	}

	out, err = execute(t, "index", "build-stats", "docs", "--engine", addr)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "vector_count:") {
		t.Errorf("build-stats output: %q", out)
	}

	if _, err = execute(t, "collection", "flush", "docs", "--engine", addr); err != nil {
		t.Fatalf("flush: %v", err)
	}

	// Drop refuses without --confirm.
	if _, err = execute(t, "collection", "drop", "docs", "--engine", addr); err == nil {
		t.Fatal("drop without --confirm must fail")
	}
	out, err = execute(t, "collection", "drop", "docs", "--engine", addr, "--confirm")
	if err != nil {
		t.Fatalf("drop: %v (%s)", err, out)
	}
	out, err = execute(t, "collection", "list", "--engine", addr)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "no collections") {
		t.Errorf("list after drop: %q", out)
	}
}

func TestCollectionCreateValidation(t *testing.T) {
	// Missing required flags fail before any dial.
	if _, err := execute(t, "collection", "create"); err == nil {
		t.Error("expected required-flag error")
	}
	if _, err := execute(t, "collection", "create", "--name", "x", "--dim", "8",
		"--metric", "cosine"); err == nil {
		t.Error("expected unsupported-metric error")
	}
}

func TestImportExportRoundTrip(t *testing.T) {
	dir := t.TempDir()
	stPath := filepath.Join(dir, "in.safetensors")
	writeTestSafeTensors(t, stPath, 6, 4)
	dataDir := filepath.Join(dir, "data")

	out, err := execute(t, "import", "--format", "safetensors",
		"--input", stPath, "--collection", "docs", "--data-dir", dataDir)
	if err != nil {
		t.Fatalf("import: %v (%s)", err, out)
	}
	if !strings.Contains(out, `imported 6 vectors into collection "docs"`) {
		t.Errorf("import output: %q", out)
	}

	outPath := filepath.Join(dir, "out.safetensors")
	if _, err := execute(t, "export", "--collection", "docs",
		"--output", outPath, "--data-dir", dataDir); err != nil {
		t.Fatalf("export: %v", err)
	}

	out, err = execute(t, "inspect", "--format", "safetensors", "--input", outPath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "1 tensors") {
		t.Errorf("inspect output: %q", out)
	}
}

func TestSyncStatusNoCheckpoint(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sync.ckpt")
	out, err := execute(t, "sync", "status", "--checkpoint", path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "no checkpoint") {
		t.Errorf("status output: %q", out)
	}
}

func TestUnknownCommandFails(t *testing.T) {
	if _, err := execute(t, "definitely-not-a-command"); err == nil {
		t.Error("expected unknown-command error")
	}
}

// writeTestSafeTensors writes an [rows, dim] F32 tensor file.
func writeTestSafeTensors(t *testing.T, path string, rows, dim int) {
	t.Helper()
	buf := make([]byte, rows*dim*4)
	for i := 0; i < rows*dim; i++ {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(float32(i%5)+0.5))
	}
	tensor := &safetensors.Tensor{
		Name: "embeddings",
		Info: safetensors.TensorInfo{Dtype: safetensors.F32, Shape: []int64{int64(rows), int64(dim)}},
		Data: buf,
	}
	var out bytes.Buffer
	if err := safetensors.Save(&out, []*safetensors.Tensor{tensor}, nil); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, out.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestAdminCommands(t *testing.T) {
	addr := startTestEngine(t)

	out, err := execute(t, "admin", "health", "--engine", addr)
	if err != nil {
		t.Fatalf("health: %v (%s)", err, out)
	}
	if !strings.Contains(out, "healthy: true") || !strings.Contains(out, "version: test") {
		t.Errorf("health output: %q", out)
	}

	out, err = execute(t, "admin", "gpu-info", "--engine", addr)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "no GPU devices") {
		t.Errorf("gpu-info output: %q", out)
	}

	// Rotator regenerate refuses locally without the exact phrase.
	if _, err := execute(t, "admin", "rotator-regenerate", "docs",
		"--engine", addr, "--confirm", "yes"); err == nil {
		t.Error("expected confirm-phrase error")
	}
}
