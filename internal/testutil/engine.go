// Package testutil provides shared helpers for tests that need a real
// engine served over gRPC.
package testutil

import (
	"net"
	"testing"

	apiv1 "github.com/gabrielleeyj/turbodb/api/v1"
	"github.com/gabrielleeyj/turbodb/internal/engine"
	"google.golang.org/grpc"
)

// StartEngine serves a real engine over gRPC on a random loopback port and
// returns its address. The engine and server are torn down via t.Cleanup.
func StartEngine(t *testing.T) string {
	t.Helper()
	addr, _ := StartEngineWithDataDir(t, t.TempDir())
	return addr
}

// StartEngineWithDataDir is like StartEngine but uses the given data
// directory, allowing tests to stop an engine and re-open the same data to
// exercise recovery. The returned stop function shuts down the server and
// engine early; it is safe to call once before the automatic cleanup.
func StartEngineWithDataDir(t *testing.T, dataDir string) (addr string, stop func()) {
	t.Helper()
	eng, err := engine.New(engine.Config{DataDir: dataDir})
	if err != nil {
		t.Fatalf("testutil: start engine: %v", err)
	}

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("testutil: listen: %v", err)
	}
	srv := grpc.NewServer()
	apiv1.RegisterTurboDBEngineServer(srv, engine.NewGRPCServer(eng))
	apiv1.RegisterTurboDBAdminServer(srv, engine.NewAdminServer(eng, "test"))
	go func() { _ = srv.Serve(lis) }()

	stopped := false
	stop = func() {
		if stopped {
			return
		}
		stopped = true
		srv.GracefulStop()
		_ = eng.Close()
	}
	t.Cleanup(stop)
	return lis.Addr().String(), stop
}
