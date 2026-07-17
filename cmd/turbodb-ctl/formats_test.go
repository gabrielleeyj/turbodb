package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestConvertSafeTensorsToGGUFRoundTrip(t *testing.T) {
	// Arrange
	dir := t.TempDir()
	stPath := filepath.Join(dir, "in.safetensors")
	writeTestSafeTensors(t, stPath, 4, 8)
	ggufPath := filepath.Join(dir, "out.gguf")

	// Act
	out, err := execute(t, "convert", "--from", "safetensors", "--to", "gguf",
		"--input", stPath, "--output", ggufPath)

	// Assert
	if err != nil {
		t.Fatalf("convert: %v (%s)", err, out)
	}
	if !strings.Contains(out, "converted") {
		t.Errorf("convert output: %q", out)
	}
	out, err = execute(t, "inspect", "--format", "gguf", "--input", ggufPath)
	if err != nil {
		t.Fatalf("inspect converted gguf: %v (%s)", err, out)
	}
	if out == "" {
		t.Error("inspect produced no output")
	}
}

func TestConvertRejectsUnsupportedFormats(t *testing.T) {
	if _, err := execute(t, "convert", "--from", "gguf", "--to", "safetensors",
		"--input", "in", "--output", "out"); err == nil ||
		!strings.Contains(err.Error(), "only --from safetensors --to gguf") {
		t.Errorf("error = %v, want unsupported-direction error", err)
	}
}

func TestImportRejectsMalformedGGUF(t *testing.T) {
	// Arrange: a file that is not GGUF at all.
	path := filepath.Join(t.TempDir(), "bogus.gguf")
	if err := os.WriteFile(path, []byte("definitely not gguf"), 0o600); err != nil {
		t.Fatal(err)
	}

	// Act
	_, err := execute(t, "import", "--format", "gguf", "--input", path,
		"--tensor", "embeddings", "--collection", "docs",
		"--data-dir", filepath.Join(t.TempDir(), "data"))

	// Assert
	if err == nil {
		t.Error("expected malformed-gguf error")
	}
}

func TestInspectRejectsMissingFile(t *testing.T) {
	if _, err := execute(t, "inspect", "--format", "safetensors",
		"--input", filepath.Join(t.TempDir(), "nope.safetensors")); err == nil {
		t.Error("expected missing-file error")
	}
}

func TestSyncReconcileRequiresDSN(t *testing.T) {
	t.Setenv("TURBODB_PG_DSN", "")
	if _, err := execute(t, "sync", "reconcile"); err == nil ||
		!strings.Contains(err.Error(), "--pg-dsn") {
		t.Errorf("error = %v, want missing-DSN error", err)
	}
}

func TestAdminCodebookUpgradeErrorPath(t *testing.T) {
	addr := startTestEngine(t)
	// The engine intentionally reports nothing to upgrade; the CLI must
	// surface that as an error, not swallow it.
	if _, err := execute(t, "admin", "codebook-upgrade", "docs",
		"--engine", addr, "--to", "v2"); err == nil {
		t.Error("expected unimplemented-upgrade error")
	}
}
