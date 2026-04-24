package index

import (
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	"github.com/gabrielleeyj/turbodb/pkg/codebook"
	"github.com/gabrielleeyj/turbodb/pkg/rotation"
)

func TestSegmentFileRoundTrip(t *testing.T) {
	rot, err := rotation.NewHadamardRotator(testDim, testSeed)
	if err != nil {
		t.Fatal(err)
	}
	cb, err := codebook.Load(testDim, testBitWidth)
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewPCG(77, 88))
	n := 50
	entries := make([]VectorEntry, n)
	for i := range n {
		entries[i] = VectorEntry{
			ID:     fmt.Sprintf("rt-vec-%d", i),
			Values: randomVec(rng, testDim),
		}
	}

	original, err := Seal("fmt-test", entries, SealedSegmentConfig{
		ID: "fmt-test", Dim: testDim, BitWidth: testBitWidth,
		Rotator: rot, Codebook: cb,
	})
	if err != nil {
		t.Fatal(err)
	}

	// Write to temp file.
	dir := t.TempDir()
	path := filepath.Join(dir, "test.tqsg")

	if err := WriteSegmentFile(path, original); err != nil {
		t.Fatalf("WriteSegmentFile: %v", err)
	}

	// Verify file was created.
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}
	if info.Size() == 0 {
		t.Fatal("segment file is empty")
	}
	t.Logf("Segment file size: %d bytes for %d vectors", info.Size(), n)

	// Read back.
	loaded, err := ReadSegmentFile(path, "fmt-test", rot, cb)
	if err != nil {
		t.Fatalf("ReadSegmentFile: %v", err)
	}

	// Verify counts match.
	if loaded.Count() != original.Count() {
		t.Fatalf("Count mismatch: loaded=%d, original=%d", loaded.Count(), original.Count())
	}

	// Verify IDs match.
	origIDs := original.IDs()
	loadedIDs := loaded.IDs()
	for i := range origIDs {
		if origIDs[i] != loadedIDs[i] {
			t.Fatalf("ID mismatch at %d: %q vs %q", i, origIDs[i], loadedIDs[i])
		}
	}

	// Verify norms match.
	origNorms := original.Norms()
	loadedNorms := loaded.Norms()
	for i := range origNorms {
		if origNorms[i] != loadedNorms[i] {
			t.Fatalf("Norm mismatch at %d: %f vs %f", i, origNorms[i], loadedNorms[i])
		}
	}

	// Verify codes match.
	origCodes := original.Codes()
	loadedCodes := loaded.Codes()
	for i := range origCodes {
		if len(origCodes[i].Indices) != len(loadedCodes[i].Indices) {
			t.Fatalf("Code length mismatch at %d: %d vs %d",
				i, len(origCodes[i].Indices), len(loadedCodes[i].Indices))
		}
		for j := range origCodes[i].Indices {
			if origCodes[i].Indices[j] != loadedCodes[i].Indices[j] {
				t.Fatalf("Code byte mismatch at vec=%d byte=%d", i, j)
			}
		}
	}

	// Verify search produces identical results.
	query := randomVec(rng, testDim)
	origResults, err := original.Search(query, 5, nil)
	if err != nil {
		t.Fatal(err)
	}
	loadedResults, err := loaded.Search(query, 5, nil)
	if err != nil {
		t.Fatal(err)
	}

	for i := range origResults {
		if origResults[i].ID != loadedResults[i].ID {
			t.Fatalf("Search result ID mismatch at %d: %q vs %q",
				i, origResults[i].ID, loadedResults[i].ID)
		}
		if origResults[i].Score != loadedResults[i].Score {
			t.Fatalf("Search result score mismatch at %d: %f vs %f",
				i, origResults[i].Score, loadedResults[i].Score)
		}
	}
}

func TestSegmentFileCRCCorruption(t *testing.T) {
	rot, _ := rotation.NewHadamardRotator(testDim, testSeed)
	cb, _ := codebook.Load(testDim, testBitWidth)

	rng := rand.New(rand.NewPCG(77, 88))
	entries := []VectorEntry{{ID: "corrupt-1", Values: randomVec(rng, testDim)}}

	seg, _ := Seal("corrupt", entries, SealedSegmentConfig{
		ID: "corrupt", Dim: testDim, BitWidth: testBitWidth,
		Rotator: rot, Codebook: cb,
	})

	dir := t.TempDir()
	path := filepath.Join(dir, "corrupt.tqsg")
	if err := WriteSegmentFile(path, seg); err != nil {
		t.Fatal(err)
	}

	// Corrupt a byte in the body.
	data, _ := os.ReadFile(path)
	data[headerSize+5] ^= 0xFF
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}

	_, err := ReadSegmentFile(path, "corrupt", rot, cb)
	if err == nil {
		t.Fatal("expected CRC error for corrupted file")
	}
}

func TestSegmentFileTooSmall(t *testing.T) {
	rot, _ := rotation.NewHadamardRotator(testDim, testSeed)
	cb, _ := codebook.Load(testDim, testBitWidth)

	dir := t.TempDir()
	path := filepath.Join(dir, "tiny.tqsg")
	if err := os.WriteFile(path, []byte("too small"), 0o644); err != nil {
		t.Fatal(err)
	}

	_, err := ReadSegmentFile(path, "tiny", rot, cb)
	if err == nil {
		t.Fatal("expected error for file too small")
	}
}
