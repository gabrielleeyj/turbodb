// TurboDB Engine is the standalone GPU-accelerated vector database server.
//
// It exposes the TurboDBEngine gRPC service over a TCP listener, persists
// collection configs and write-ahead log records to a data directory, and
// recovers state on restart by replaying the WAL onto in-memory indexes.
//
// A second HTTP listener exposes Prometheus metrics at /metrics and a
// liveness probe at /healthz.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	apiv1 "github.com/gabrielleeyj/turbodb/api/v1"
	"github.com/gabrielleeyj/turbodb/internal/engine"
	"github.com/gabrielleeyj/turbodb/pkg/telemetry"
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
		listen        = flag.String("listen", ":7080", "gRPC listen address (host:port)")
		metricsListen = flag.String("metrics-listen", ":9090", "Prometheus + healthz HTTP listen address (empty disables)")
		dataDir       = flag.String("data-dir", "./turbodb-data", "directory for collection configs, WAL, and segments")
		logLevel      = flag.String("log-level", "info", "log level: debug, info, warn, error")
		logFormat     = flag.String("log-format", "json", "log format: json or text")
	)
	flag.Parse()

	logger := telemetry.NewLogger(os.Stderr, telemetry.LogFormat(*logFormat), *logLevel)

	// Engine first; pass it as the StatsSource for metrics so the gauges
	// reflect live state. Metrics are wired back in after construction.
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

	metrics, err := telemetry.New(telemetry.Options{Source: eng})
	if err != nil {
		return fmt.Errorf("init metrics: %w", err)
	}
	eng.AttachMetrics(metrics)

	lis, err := net.Listen("tcp", *listen)
	if err != nil {
		return fmt.Errorf("listen %s: %w", *listen, err)
	}

	server := grpc.NewServer()
	apiv1.RegisterTurboDBEngineServer(server, engine.NewGRPCServer(eng))

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	var metricsSrv *http.Server
	if *metricsListen != "" {
		mux := http.NewServeMux()
		mux.Handle("/metrics", metrics.Handler())
		mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte("ok"))
		})
		metricsSrv = &http.Server{
			Addr:              *metricsListen,
			Handler:           mux,
			ReadHeaderTimeout: 5 * time.Second,
		}
		go func() {
			logger.Info("turbodb-engine: metrics serving", "address", *metricsListen)
			if err := metricsSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
				logger.Error("metrics server", "error", err)
			}
		}()
	}

	go func() {
		<-ctx.Done()
		logger.Info("turbodb-engine: shutdown signal received")
		if metricsSrv != nil {
			shutdownCtx, c := context.WithTimeout(context.Background(), 5*time.Second)
			defer c()
			_ = metricsSrv.Shutdown(shutdownCtx)
		}
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
