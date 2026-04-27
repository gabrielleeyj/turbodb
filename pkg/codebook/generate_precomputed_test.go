package codebook

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// TestGeneratePrecomputed generates precomputed codebook JSON files for all
// standard (d, b) pairs. Run with:
//
//	GENERATE_CODEBOOKS=1 go test -run TestGeneratePrecomputed -v ./pkg/codebook/

func TestGeneratePrecomputed(t *testing.T) {
	if os.Getenv("GENERATE_CODEBOOKS") == "" {
		t.Skip("set GENERATE_CODEBOOKS=1 to regenerate precomputed codebooks")
	}

	dims := []int{128, 256, 512, 768, 1024, 1536, 3072, 4096}
	bitWidths := []int{1, 2, 3, 4, 5, 6, 8}

	// Find the precomputed directory relative to this test file.
	_, thisFile, _, _ := runtime.Caller(0)
	dir := filepath.Join(filepath.Dir(thisFile), "precomputed")
	if err := os.MkdirAll(dir, 0o750); err != nil {
		t.Fatalf("failed to create precomputed dir: %v", err)
	}

	for _, d := range dims {
		for _, b := range bitWidths {
			t.Logf("generating d=%d b=%d", d, b)
			cb, err := Generate(context.Background(), d, b)
			if err != nil {
				t.Fatalf("failed to generate d=%d b=%d: %v", d, b, err)
			}

			entry := precomputedEntry{
				Dim:       d,
				BitWidth:  b,
				Centroids: cb.Centroids(),
			}

			data, err := json.MarshalIndent(entry, "", "  ")
			if err != nil {
				t.Fatalf("failed to marshal d=%d b=%d: %v", d, b, err)
			}

			filename := filepath.Join(dir, cacheKey(d, b)+".json")
			if err := os.WriteFile(filename, data, 0o600); err != nil {
				t.Fatalf("failed to write %s: %v", filename, err)
			}
		}
	}

	t.Logf("generated %d codebook files in %s", len(dims)*len(bitWidths), dir)
}
