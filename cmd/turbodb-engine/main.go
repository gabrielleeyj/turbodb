// TurboDB Engine is the standalone GPU-accelerated vector database server.
//
// It exposes the TurboDBEngine gRPC service over a TCP listener, persists
// collection configs and write-ahead log records to a data directory, and
// recovers state on restart by replaying the WAL onto in-memory indexes.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"os"
	"os/signal"
	"syscall"

	apiv1 "github.com/gabrielleeyj/turbodb/api/v1"
	"github.com/gabrielleeyj/turbodb/internal/engine"
	"google.golang.org/grpc"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "turbodb-engine: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	var (
		listen   = flag.String("listen", ":7080", "gRPC listen address (host:port)")
		dataDir  = flag.String("data-dir", "./turbodb-data", "directory for collection configs, WAL, and segments")
		logLevel = flag.String("log-level", "info", "log level: debug, info, warn, error")
	)
	flag.Parse()

	logger := newLogger(*logLevel)
	slog.SetDefault(logger)

	eng, err := engine.New(engine.EngineConfig{
		DataDir: *dataDir,
		Logger:  logger,
	})
	if err != nil {
		return fmt.Errorf("init engine: %w", err)
	}
	defer func() {
		if cerr := eng.Close(); cerr != nil {
			logger.Error("engine close", "error", cerr)
		}
	}()

	lis, err := net.Listen("tcp", *listen)
	if err != nil {
		return fmt.Errorf("listen %s: %w", *listen, err)
	}

	server := grpc.NewServer()
	apiv1.RegisterTurboDBEngineServer(server, engine.NewGRPCServer(eng))

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	go func() {
		<-ctx.Done()
		logger.Info("turbodb-engine: shutdown signal received")
		server.GracefulStop()
	}()

	logger.Info("turbodb-engine: serving",
		"address", lis.Addr().String(),
		"data_dir", *dataDir,
	)
	if err := server.Serve(lis); err != nil {
		return fmt.Errorf("serve: %w", err)
	}
	logger.Info("turbodb-engine: stopped")
	return nil
}

func newLogger(level string) *slog.Logger {
	var lvl slog.Level
	switch level {
	case "debug":
		lvl = slog.LevelDebug
	case "warn":
		lvl = slog.LevelWarn
	case "error":
		lvl = slog.LevelError
	default:
		lvl = slog.LevelInfo
	}
	return slog.New(slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{Level: lvl}))
}
