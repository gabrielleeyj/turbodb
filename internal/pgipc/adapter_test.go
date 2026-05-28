package pgipc

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/gabrielleeyj/turbodb/internal/engine"
	"github.com/gabrielleeyj/turbodb/internal/pgproto"
)

// TestAdapterEndToEnd drives a real engine through the IPC server and Go client:
// BUILD_BEGIN -> BUILD_VECTOR* -> BUILD_COMMIT -> SEARCH -> STATS, mirroring the
// pg_turboquant access-method flow.
func TestAdapterEndToEnd(t *testing.T) {
	eng, err := engine.New(engine.EngineConfig{DataDir: filepath.Join(t.TempDir(), "data")})
	if err != nil {
		t.Fatal(err)
	}
	defer eng.Close()

	sock := filepath.Join(os.TempDir(), fmt.Sprintf("tq-ipc-%d.sock", time.Now().UnixNano()))
	t.Cleanup(func() { os.Remove(sock) })
	srv := pgproto.NewServer(NewAdapter(eng), pgproto.ServerConfig{SocketPath: sock, AllowedUID: -1})
	if err := srv.Listen(); err != nil {
		t.Fatal(err)
	}
	go srv.Serve(context.Background())
	t.Cleanup(func() { srv.Close() })

	c := dialWithRetry(t, sock)
	defer c.Close()

	const dim = 8
	if err := c.BuildBegin(pgproto.BuildBegin{Collection: "idx", Dim: dim, BitWidth: 4}); err != nil {
		t.Fatal(err)
	}
	// Insert 20 distinct vectors via BUILD_VECTOR.
	vecs := make([][]float32, 20)
	for i := range vecs {
		v := make([]float32, dim)
		for j := range v {
			v[j] = float32((i + j) % 5)
		}
		vecs[i] = v
		if err := c.BuildVector(pgproto.VectorMsg{TID: uint64(i + 1), Values: v}); err != nil {
			t.Fatalf("build vector %d: %v", i, err)
		}
	}
	if err := c.BuildCommit(); err != nil {
		t.Fatal(err)
	}

	// Query with the first vector; the top result should be its TID (1).
	results, err := c.Search(pgproto.SearchBegin{Collection: "idx", Query: vecs[0], TopK: 5, OversearchFactor: 2.0})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) == 0 {
		t.Fatal("no search results")
	}
	if results[0].TID != 1 {
		t.Errorf("top result TID = %d, want 1", results[0].TID)
	}

	// Delete TID 1, then stats should reflect the tombstone count via vector
	// count semantics.
	if err := c.Delete(pgproto.DeleteMsg{TID: 1}); err != nil {
		t.Fatal(err)
	}
	stats, err := c.Stats()
	if err != nil {
		t.Fatal(err)
	}
	if stats.VectorCount == 0 {
		t.Errorf("stats vector count = 0, expected > 0")
	}
}

func dialWithRetry(t *testing.T, sock string) *pgproto.Client {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		c, err := pgproto.Dial(sock)
		if err == nil {
			return c
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("could not connect to %s", sock)
	return nil
}

func TestAdapterIdempotentBuildBeginAndErrors(t *testing.T) {
	eng, err := engine.New(engine.EngineConfig{DataDir: filepath.Join(t.TempDir(), "data")})
	if err != nil {
		t.Fatal(err)
	}
	defer eng.Close()
	a := NewAdapter(eng)
	ctx := context.Background()

	bb := pgproto.BuildBegin{Collection: "c", Dim: 4, BitWidth: 4}
	if err := a.BuildBegin(ctx, bb); err != nil {
		t.Fatal(err)
	}
	// Second BuildBegin on the same collection must be a no-op, not an error.
	if err := a.BuildBegin(ctx, bb); err != nil {
		t.Errorf("idempotent BuildBegin failed: %v", err)
	}

	// Stats / Search on a missing collection must error.
	if _, err := a.Stats(ctx, "missing"); err == nil {
		t.Error("expected stats error for missing collection")
	}
	if _, err := a.Search(ctx, pgproto.SearchBegin{Collection: "missing", Query: []float32{1, 2, 3, 4}, TopK: 1}); err == nil {
		t.Error("expected search error for missing collection")
	}
}

func TestAdapterTIDMapping(t *testing.T) {
	if tidToID(42) != "42" {
		t.Errorf("tidToID(42) = %q", tidToID(42))
	}
	if idToTID("42") != 42 {
		t.Errorf("idToTID(42) = %d", idToTID("42"))
	}
	if idToTID("not-a-number") != 0 {
		t.Errorf("idToTID(non-numeric) should be 0")
	}
}
