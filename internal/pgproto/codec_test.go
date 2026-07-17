package pgproto

import (
	"encoding/binary"
	"os"
	"runtime"
	"testing"
	"time"
)

// TestDecodeTruncatedPayloads verifies that every strict prefix of a valid
// encoding returns an error rather than panicking or silently succeeding.
func TestDecodeTruncatedPayloads(t *testing.T) {
	tests := []struct {
		name    string
		encoded []byte
		decode  func([]byte) error
	}{
		{
			name: "BuildBegin",
			encoded: BuildBegin{
				Collection: "docs", Dim: 128, BitWidth: 4, Variant: 0, RotatorSeed: 7,
			}.Encode(),
			decode: func(p []byte) error { _, err := DecodeBuildBegin(p); return err },
		},
		{
			name:    "VectorMsg",
			encoded: VectorMsg{TID: 42, Values: []float32{1, 2, 3}}.Encode(),
			decode:  func(p []byte) error { _, err := DecodeVectorMsg(p); return err },
		},
		{
			name:    "DeleteMsg",
			encoded: DeleteMsg{TID: 42}.Encode(),
			decode:  func(p []byte) error { _, err := DecodeDeleteMsg(p); return err },
		},
		{
			name: "SearchBegin",
			encoded: SearchBegin{
				Collection: "docs", Query: []float32{1, 2}, TopK: 5,
				OversearchFactor: 2, Rerank: true, Exact: true,
			}.Encode(),
			decode: func(p []byte) error { _, err := DecodeSearchBegin(p); return err },
		},
		{
			name:    "ResultMsg",
			encoded: ResultMsg{TID: 9, Score: 0.5, Done: true}.Encode(),
			decode:  func(p []byte) error { _, err := DecodeResultMsg(p); return err },
		},
		{
			name: "StatsReply",
			encoded: StatsReply{
				VectorCount: 10, SealedSegments: 1, GrowingSegment: 1, PinnedBytes: 64,
			}.Encode(),
			decode: func(p []byte) error { _, err := DecodeStatsReply(p); return err },
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Sanity: the full encoding decodes cleanly.
			if err := tt.decode(tt.encoded); err != nil {
				t.Fatalf("full payload failed to decode: %v", err)
			}
			// Every strict prefix must error.
			for n := 0; n < len(tt.encoded); n++ {
				if err := tt.decode(tt.encoded[:n]); err == nil {
					t.Errorf("prefix of %d/%d bytes decoded without error", n, len(tt.encoded))
				}
			}
		})
	}
}

// TestDecodeAdversarialLengthPrefixes feeds huge declared lengths with tiny
// bodies; decoders must fail via bounds checks, not allocate or panic.
func TestDecodeAdversarialLengthPrefixes(t *testing.T) {
	huge := make([]byte, 4)
	binary.LittleEndian.PutUint32(huge, 0xFFFFFFFF)

	t.Run("string length overflows payload", func(t *testing.T) {
		// BuildBegin starts with a length-prefixed collection name.
		if _, err := DecodeBuildBegin(huge); err == nil {
			t.Error("expected error for 4GiB string claim")
		}
	})
	t.Run("vector length overflows payload", func(t *testing.T) {
		// VectorMsg: 8-byte TID then a length-prefixed vector.
		payload := append(make([]byte, 8), huge...)
		if _, err := DecodeVectorMsg(payload); err == nil {
			t.Error("expected error for 4G-element vector claim")
		}
	})
	t.Run("vector length with partial data", func(t *testing.T) {
		payload := append(make([]byte, 8), huge...)
		payload = append(payload, make([]byte, 16)...) // 4 floats, far fewer than claimed
		if _, err := DecodeVectorMsg(payload); err == nil {
			t.Error("expected error for undersized vector body")
		}
	})
	t.Run("search query length overflows", func(t *testing.T) {
		w := writer{}
		w.str("docs")
		payload := append(w.buf, huge...)
		if _, err := DecodeSearchBegin(payload); err == nil {
			t.Error("expected error for oversized query claim")
		}
	})
}

// TestDecodeErrorTruncated ensures the error-payload decoder tolerates
// truncated input (it returns an empty string rather than panicking).
func TestDecodeErrorTruncated(t *testing.T) {
	if got := DecodeError([]byte{0xFF, 0xFF, 0xFF, 0xFF}); got != "" {
		t.Errorf("truncated error payload decoded to %q, want empty", got)
	}
}

// TestPeerCredAuthorization verifies the SO_PEERCRED/LOCAL_PEERCRED check:
// the owning UID is accepted and any other UID is rejected before frame
// handling.
func TestPeerCredAuthorization(t *testing.T) {
	if runtime.GOOS != "linux" && runtime.GOOS != "darwin" {
		t.Skipf("peer credentials unsupported on %s", runtime.GOOS)
	}

	t.Run("matching uid accepted", func(t *testing.T) {
		// Arrange
		h := &fakeHandler{}
		sock := startTestServerWithUID(t, h, os.Getuid())
		c, err := Dial(sock)
		if err != nil {
			t.Fatal(err)
		}
		defer func() { _ = c.Close() }()

		// Act
		err = c.BuildBegin(BuildBegin{Collection: "docs", Dim: 4, BitWidth: 4})

		// Assert
		if err != nil {
			t.Fatalf("expected authorized request to succeed: %v", err)
		}
	})

	t.Run("mismatched uid rejected", func(t *testing.T) {
		// Arrange
		h := &fakeHandler{}
		sock := startTestServerWithUID(t, h, os.Getuid()+1)

		// Act: the server closes the connection during authorize, so either
		// the dial or the first round-trip must fail.
		c, err := Dial(sock)
		if err == nil {
			defer func() { _ = c.Close() }()
			err = c.BuildBegin(BuildBegin{Collection: "docs", Dim: 4, BitWidth: 4})
		}

		// Assert
		if err == nil {
			t.Fatal("expected rejected connection to fail")
		}
		h.mu.Lock()
		defer h.mu.Unlock()
		if len(h.inserts) != 0 {
			t.Error("handler must not observe traffic from rejected peers")
		}
	})
}

// startTestServerWithUID is startTestServer with an explicit AllowedUID and
// without the connectability pre-check (which would itself be rejected).
func startTestServerWithUID(t *testing.T, h Handler, uid int) string {
	t.Helper()
	sock := shortSocketPath(t)
	srv := NewServer(h, ServerConfig{SocketPath: sock, AllowedUID: uid})
	if err := srv.Listen(); err != nil {
		t.Fatal(err)
	}
	go func() { _ = srv.Serve(t.Context()) }()
	t.Cleanup(func() { _ = srv.Close() })
	// Wait until the socket file exists so Dial does not race Listen.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(sock); err == nil {
			return sock
		}
		time.Sleep(2 * time.Millisecond)
	}
	t.Fatalf("socket %s never appeared", sock)
	return ""
}

// shortSocketPath returns a unique socket path under the system temp dir,
// keeping within the ~104-byte unix socket path limit on macOS.
func shortSocketPath(t *testing.T) string {
	t.Helper()
	f, err := os.CreateTemp("", "tqp-*.sock")
	if err != nil {
		t.Fatal(err)
	}
	path := f.Name()
	_ = f.Close()
	_ = os.Remove(path)
	t.Cleanup(func() { _ = os.Remove(path) })
	return path
}
