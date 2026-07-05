package main

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"time"

	apiv1 "github.com/gabrielleeyj/turbodb/api/v1"
	"github.com/gabrielleeyj/turbodb/pkg/replication"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// Fixed topology for the soak stack. The engine data dir, sync checkpoint,
// and PostgreSQL state all persist across injected crashes.
const (
	soakCollection = "soak"
	soakTable      = "public.soak_docs"
	soakDim        = 8
	engineGRPC     = "127.0.0.1:17100"
	engineAdmin    = "127.0.0.1:18100"
	soakSlot       = "turbodb_soak"
	soakPub        = "turbodb_soak_pub"
)

type soak struct {
	cfg    soakConfig
	logger *slog.Logger
	board  *scoreboard

	engine   *managedProc
	sync     *managedProc
	workload *workload
	ckptPath string
}

// newSoak prepares PostgreSQL (table, publication), writes sync.yaml, starts
// the engine and sync under supervision, and creates the collection.
func newSoak(ctx context.Context, cfg soakConfig, logger *slog.Logger) (*soak, error) {
	s := &soak{cfg: cfg, logger: logger, board: newScoreboard()}
	s.ckptPath = filepath.Join(cfg.workdir, "sync.ckpt")

	if err := s.setupPostgres(ctx); err != nil {
		return nil, err
	}

	syncYAML := filepath.Join(cfg.workdir, "sync.yaml")
	spec := fmt.Sprintf(`tables:
  - postgres: %s
    engine:   %s
    columns:
      id:        doc_id
      embedding: embedding
    filter:    "deleted_at IS NULL"
`, soakTable, soakCollection)
	if err := os.WriteFile(syncYAML, []byte(spec), 0o600); err != nil {
		return nil, err
	}

	s.engine = newManagedProc("engine", logger, s.board, 0, cfg.engineBin,
		"--listen", engineGRPC,
		"--admin-listen", engineAdmin,
		"--metrics-listen", "",
		"--data-dir", filepath.Join(cfg.workdir, "engine-data"),
		"--log-format", "text",
		"--wal-fsync", "group",
	)
	s.engine.logTo(filepath.Join(cfg.workdir, "engine.log"))
	s.engine.start()
	if err := s.waitEngineReady(ctx, 30*time.Second); err != nil {
		return nil, err
	}
	if err := s.ensureCollection(ctx); err != nil {
		return nil, err
	}

	s.sync = newManagedProc("sync", logger, s.board, 2*time.Second, cfg.syncBin,
		"run",
		"--config", syncYAML,
		"--pg-dsn", cfg.pgDSN,
		"--slot", soakSlot,
		"--publication", soakPub,
		"--engine", engineGRPC,
		"--checkpoint", s.ckptPath,
	)
	// Sync exits by design when the engine is unreachable (circuit breaker)
	// or the replication connection drops; the supervisor restarting it is
	// the production model (systemd Restart=always), not a violation.
	s.sync.exitsAreExpected = true
	s.sync.logTo(filepath.Join(cfg.workdir, "sync.log"))
	s.sync.start()

	// Rows written before sync creates its replication slot are never
	// streamed, so hold the workload (and the first sentinel) until the
	// slot is live.
	if err := s.waitSlotActive(ctx, 30*time.Second); err != nil {
		return nil, err
	}

	s.workload = newWorkload(cfg.pgDSN, cfg.workloadRate, logger)
	s.workload.start(ctx)
	return s, nil
}

// waitSlotActive polls until sync's replication slot exists and has an
// attached walsender.
func (s *soak) waitSlotActive(ctx context.Context, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	query := fmt.Sprintf("SELECT active FROM pg_replication_slots WHERE slot_name = '%s'", soakSlot)
	for time.Now().Before(deadline) && ctx.Err() == nil {
		if v, err := s.pgQueryOne(ctx, query); err == nil && v == "t" {
			return nil
		}
		time.Sleep(300 * time.Millisecond)
	}
	return fmt.Errorf("replication slot %s not active within %s", soakSlot, timeout)
}

func (s *soak) shutdown() {
	if s.workload != nil {
		s.workload.stop()
	}
	if s.sync != nil {
		s.sync.stop()
	}
	if s.engine != nil {
		s.engine.stop()
	}
}

// waitEngineReady polls the admin /readyz probe.
func (s *soak) waitEngineReady(ctx context.Context, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	url := "http://" + engineAdmin + "/readyz"
	for time.Now().Before(deadline) && ctx.Err() == nil {
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		resp, err := http.DefaultClient.Do(req)
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		time.Sleep(300 * time.Millisecond)
	}
	return fmt.Errorf("engine not ready within %s", timeout)
}

// ensureCollection creates the soak collection if it does not exist yet
// (it persists across engine restarts within one workdir).
func (s *soak) ensureCollection(ctx context.Context) error {
	conn, err := grpc.NewClient(engineGRPC, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return err
	}
	defer func() { _ = conn.Close() }()
	client := apiv1.NewTurboDBEngineClient(conn)

	list, err := client.ListCollections(ctx, &apiv1.ListCollectionsRequest{})
	if err != nil {
		return fmt.Errorf("list collections: %w", err)
	}
	for _, c := range list.GetCollections() {
		if c.GetName() == soakCollection {
			return nil
		}
	}
	_, err = client.CreateCollection(ctx, &apiv1.CreateCollectionRequest{
		Config: &apiv1.CollectionConfig{
			Name:      soakCollection,
			Dimension: soakDim,
			BitWidth:  4,
			Metric:    apiv1.Metric_METRIC_INNER_PRODUCT,
			Variant:   apiv1.QuantizationVariant_QUANTIZATION_VARIANT_MSE,
		},
	})
	if err != nil {
		return fmt.Errorf("create collection: %w", err)
	}
	return nil
}

// checkpointLSN reads sync's persisted position; 0 means no checkpoint yet.
func (s *soak) checkpointLSN() (uint64, error) {
	cp, err := replication.NewFileCheckpoint(s.ckptPath)
	if err != nil {
		return 0, err
	}
	return cp.Load()
}
