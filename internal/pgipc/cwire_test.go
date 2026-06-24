package pgipc

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/gabrielleeyj/turbodb/internal/engine"
	"github.com/gabrielleeyj/turbodb/internal/pgproto"
)

// TestCClientWireCompatibility compiles the C IPC client probe and runs it
// against the Go server backed by a real engine, proving the C and Go protocol
// implementations are wire-compatible. Skipped when no C compiler is available.
func TestCClientWireCompatibility(t *testing.T) {
	cc, err := exec.LookPath("cc")
	if err != nil {
		t.Skip("no C compiler available")
	}

	repoRoot, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	pgDir := filepath.Join(repoRoot, "postgres")
	bin := filepath.Join(t.TempDir(), "probe")
	build := exec.Command(cc, "-std=c11", "-O2",
		filepath.Join(pgDir, "turbodb_ipc.c"),
		filepath.Join(pgDir, "turbodb_ipc_probe.c"),
		"-I", pgDir, "-o", bin)
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("compile probe: %v\n%s", err, out)
	}

	// Start the engine + IPC server.
	eng, err := engine.New(engine.EngineConfig{DataDir: filepath.Join(t.TempDir(), "data")})
	if err != nil {
		t.Fatal(err)
	}
	defer eng.Close()
	sock := filepath.Join(os.TempDir(), fmt.Sprintf("tq-cwire-%d.sock", time.Now().UnixNano()))
	t.Cleanup(func() { os.Remove(sock) })
	srv := pgproto.NewServer(NewAdapter(eng), pgproto.ServerConfig{SocketPath: sock, AllowedUID: -1})
	if err := srv.Listen(); err != nil {
		t.Fatal(err)
	}
	go srv.Serve(context.Background())
	t.Cleanup(func() { srv.Close() })
	// Wait for the socket.
	dialWithRetry(t, sock).Close()

	out, err := exec.Command(bin, sock).CombinedOutput()
	if err != nil {
		t.Fatalf("probe run: %v\n%s", err, out)
	}
	got := strings.TrimSpace(string(out))
	if !strings.HasPrefix(got, "OK ") {
		t.Fatalf("probe output = %q", got)
	}
	// Top result for a self-query should be tid=1; 10 vectors inserted.
	if !strings.Contains(got, "tid=1 ") || !strings.Contains(got, "vcount=10") {
		t.Errorf("unexpected probe output: %q", got)
	}
}
