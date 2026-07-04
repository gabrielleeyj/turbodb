package pgproto

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
	"time"
)

func TestFrameRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	payload := []byte("hello payload")
	if err := WriteFrame(&buf, OpInsert, payload); err != nil {
		t.Fatal(err)
	}
	f, err := ReadFrame(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if f.Opcode != OpInsert || !bytes.Equal(f.Payload, payload) {
		t.Fatalf("frame = %+v", f)
	}
}

func TestFrameRejectsBadVersion(t *testing.T) {
	// Hand-build a frame with a wrong schema version.
	var buf bytes.Buffer
	body := []byte{0, byte(OpInsert), 0x99, 0x00} // opcode + bad version (LE)
	lenPrefix := []byte{0, 0, 0, byte(len(body))}
	buf.Write(lenPrefix)
	buf.Write(body)
	if _, err := ReadFrame(&buf); err == nil {
		t.Error("expected schema version error")
	}
}

func TestMessageCodecs(t *testing.T) {
	t.Run("BuildBegin", func(t *testing.T) {
		m := BuildBegin{Collection: "docs", Dim: 128, BitWidth: 4, Variant: 0, RotatorSeed: 99}
		got, err := DecodeBuildBegin(m.Encode())
		if err != nil || !reflect.DeepEqual(got, m) {
			t.Fatalf("got %+v err %v", got, err)
		}
	})
	t.Run("VectorMsg", func(t *testing.T) {
		m := VectorMsg{TID: 42, Values: []float32{1, 2, 3, -4.5}}
		got, err := DecodeVectorMsg(m.Encode())
		if err != nil || !reflect.DeepEqual(got, m) {
			t.Fatalf("got %+v err %v", got, err)
		}
	})
	t.Run("SearchBegin", func(t *testing.T) {
		m := SearchBegin{Collection: "c", Query: []float32{0.5, 0.25}, TopK: 10, OversearchFactor: 2.0, Rerank: true, Exact: false}
		got, err := DecodeSearchBegin(m.Encode())
		if err != nil || !reflect.DeepEqual(got, m) {
			t.Fatalf("got %+v err %v", got, err)
		}
	})
	t.Run("ResultMsg", func(t *testing.T) {
		m := ResultMsg{TID: 7, Score: 1.25, Done: false}
		got, err := DecodeResultMsg(m.Encode())
		if err != nil || got != m {
			t.Fatalf("got %+v err %v", got, err)
		}
	})
	t.Run("StatsReply", func(t *testing.T) {
		m := StatsReply{VectorCount: 1000, SealedSegments: 2, GrowingSegment: 1, PinnedBytes: 4096}
		got, err := DecodeStatsReply(m.Encode())
		if err != nil || got != m {
			t.Fatalf("got %+v err %v", got, err)
		}
	})
}

func TestCodecTruncated(t *testing.T) {
	if _, err := DecodeVectorMsg([]byte{1, 2}); err == nil {
		t.Error("expected truncation error")
	}
}

// fakeHandler records operations and returns canned search results.
type fakeHandler struct {
	mu        sync.Mutex
	inserts   []VectorMsg
	deletes   []DeleteMsg
	committed []string
	results   []ResultMsg
}

func (h *fakeHandler) BuildBegin(_ context.Context, _ BuildBegin) error { return nil }
func (h *fakeHandler) Insert(_ context.Context, _ string, m VectorMsg) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.inserts = append(h.inserts, m)
	return nil
}
func (h *fakeHandler) Delete(_ context.Context, _ string, m DeleteMsg) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.deletes = append(h.deletes, m)
	return nil
}
func (h *fakeHandler) Commit(_ context.Context, c string) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.committed = append(h.committed, c)
	return nil
}
func (h *fakeHandler) Search(_ context.Context, _ SearchBegin) ([]ResultMsg, error) {
	return h.results, nil
}
func (h *fakeHandler) Stats(_ context.Context, _ string) (StatsReply, error) {
	return StatsReply{VectorCount: uint64(len(h.inserts))}, nil
}

func startTestServer(t *testing.T, h Handler) string {
	t.Helper()
	// Unix socket paths are capped (~104 bytes on macOS); use a short path
	// under the system temp dir rather than the long per-test temp dir.
	sock := filepath.Join(os.TempDir(), fmt.Sprintf("tq-%d.sock", time.Now().UnixNano()))
	t.Cleanup(func() { _ = os.Remove(sock) })
	srv := NewServer(h, ServerConfig{SocketPath: sock, AllowedUID: -1})
	if err := srv.Listen(); err != nil {
		t.Fatal(err)
	}
	go func() { _ = srv.Serve(context.Background()) }()
	// Wait for the socket to be connectable.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if c, err := Dial(sock); err == nil {
			_ = c.Close()
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Cleanup(func() { _ = srv.Close() })
	return sock
}

func TestClientServerFlow(t *testing.T) {
	h := &fakeHandler{results: []ResultMsg{{TID: 1, Score: 0.9}, {TID: 2, Score: 0.8}}}
	sock := startTestServer(t, h)

	c, err := Dial(sock)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()

	if err := c.BuildBegin(BuildBegin{Collection: "docs", Dim: 4, BitWidth: 4}); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 3; i++ {
		if err := c.BuildVector(VectorMsg{TID: uint64(i), Values: []float32{1, 2, 3, 4}}); err != nil {
			t.Fatal(err)
		}
	}
	if err := c.BuildCommit(); err != nil {
		t.Fatal(err)
	}
	if err := c.Delete(DeleteMsg{TID: 1}); err != nil {
		t.Fatal(err)
	}

	results, err := c.Search(SearchBegin{Collection: "docs", Query: []float32{1, 2, 3, 4}, TopK: 2})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 || results[0].TID != 1 || results[1].TID != 2 {
		t.Fatalf("search results = %+v", results)
	}

	stats, err := c.Stats()
	if err != nil {
		t.Fatal(err)
	}
	if stats.VectorCount != 3 {
		t.Errorf("stats vector count = %d", stats.VectorCount)
	}

	h.mu.Lock()
	defer h.mu.Unlock()
	if len(h.inserts) != 3 || len(h.deletes) != 1 || len(h.committed) != 1 {
		t.Errorf("handler state: inserts=%d deletes=%d committed=%d", len(h.inserts), len(h.deletes), len(h.committed))
	}
}

func TestServerRequiresActiveCollection(t *testing.T) {
	sock := startTestServer(t, &fakeHandler{})
	c, err := Dial(sock)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()
	// Insert before any BuildBegin/SearchBegin must error.
	if err := c.Insert(VectorMsg{TID: 1, Values: []float32{1}}); err == nil {
		t.Error("expected no-active-collection error")
	}
}

// errHandler returns an error from Search and Insert to exercise the error
// reply path.
type errHandler struct{ fakeHandler }

func (h *errHandler) Search(_ context.Context, _ SearchBegin) ([]ResultMsg, error) {
	return nil, fmt.Errorf("boom")
}

func TestServerErrorReply(t *testing.T) {
	sock := startTestServer(t, &errHandler{})
	c, err := Dial(sock)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()
	if _, err := c.Search(SearchBegin{Collection: "c", Query: []float32{1}, TopK: 1}); err == nil || err.Error() != "boom" {
		t.Fatalf("expected boom error, got %v", err)
	}
}

func TestOpcodeStringAndAddr(t *testing.T) {
	for _, op := range []Opcode{OpBuildBegin, OpInsert, OpSearchNext, OpAck, OpError, OpResult, Opcode(9999)} {
		if op.String() == "" {
			t.Errorf("empty string for opcode %d", op)
		}
	}
	srv := NewServer(&fakeHandler{}, ServerConfig{SocketPath: "/tmp/x.sock", AllowedUID: -1})
	if srv.Addr() != "/tmp/x.sock" {
		t.Errorf("Addr = %q", srv.Addr())
	}
}

func TestFrameTooLarge(t *testing.T) {
	var buf bytes.Buffer
	// Length prefix claiming an enormous body.
	buf.Write([]byte{0xff, 0xff, 0xff, 0xff})
	if _, err := ReadFrame(&buf); err == nil {
		t.Error("expected frame-too-large error")
	}
	big := make([]byte, MaxFrameBytes+1)
	if err := WriteFrame(&bytes.Buffer{}, OpInsert, big); err == nil {
		t.Error("expected write frame-too-large error")
	}
}

func TestErrorPayloadRoundTrip(t *testing.T) {
	if got := DecodeError(EncodeError("oops")); got != "oops" {
		t.Errorf("error payload = %q", got)
	}
}

func TestShutdown(t *testing.T) {
	sock := startTestServer(t, &fakeHandler{})
	c, err := Dial(sock)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = c.Close() }()
	if err := c.Shutdown(); err != nil {
		t.Fatal(err)
	}
}
