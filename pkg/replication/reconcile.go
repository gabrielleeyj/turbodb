package replication

import (
	"context"
	"fmt"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// SourceScanner streams rows from the source-of-truth table in bytewise
// ascending order of the mapped id column (rendered as text). PgTableScanner
// implements this against PostgreSQL; tests use fakes.
type SourceScanner interface {
	// ScanRows returns up to limit rows whose id (as text) is strictly
	// greater than afterID, ordered bytewise ascending.
	ScanRows(ctx context.Context, mapping TableMapping, afterID string, limit int) (rows []map[string]any, hasMore bool, err error)
}

// IndexIDLister streams the engine collection's live ids in bytewise
// ascending order. The gRPC ListIDs RPC implements this.
type IndexIDLister interface {
	ListIDs(ctx context.Context, collection, afterID string, pageSize int) (ids []string, hasMore bool, err error)
}

// ReconcileReport summarizes one reconciliation pass over one table mapping.
type ReconcileReport struct {
	Collection string
	// SourceRows is the number of source rows scanned (matching the filter).
	SourceRows int
	// EngineIDs is the number of live ids scanned from the engine.
	EngineIDs int
	// MissingInEngine are ids present in the source but absent from the
	// engine (repair: upsert).
	MissingInEngine []string
	// OrphanedInEngine are ids present in the engine but absent from the
	// source, or filtered out of it (repair: delete).
	OrphanedInEngine []string
	// MalformedRows counts source rows that could not be transformed.
	MalformedRows int
	// Repaired is true when repair ops were applied to the engine.
	Repaired bool
	// Duration is the wall time of the pass.
	Duration time.Duration
}

// Discrepancies returns the total number of mismatches found.
func (r ReconcileReport) Discrepancies() int {
	return len(r.MissingInEngine) + len(r.OrphanedInEngine)
}

// ReconcileMetrics holds the Prometheus instruments for reconciliation runs
// (SCOPE Task 7.4).
type ReconcileMetrics struct {
	discrepancies *prometheus.CounterVec
	lastRun       *prometheus.GaugeVec
}

// NewReconcileMetrics registers the reconciliation metrics on reg.
func NewReconcileMetrics(reg prometheus.Registerer) *ReconcileMetrics {
	m := &ReconcileMetrics{
		discrepancies: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "turbodb_sync_reconcile_discrepancies_total",
			Help: "Mismatches found between the source table and the engine collection.",
		}, []string{"collection", "kind"}),
		lastRun: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "turbodb_sync_reconcile_last_run_seconds",
			Help: "Duration in seconds of the last completed reconciliation pass.",
		}, []string{"collection"}),
	}
	reg.MustRegister(m.discrepancies, m.lastRun)
	return m
}

func (m *ReconcileMetrics) observe(r ReconcileReport) {
	if m == nil {
		return
	}
	m.discrepancies.WithLabelValues(r.Collection, "missing").Add(float64(len(r.MissingInEngine)))
	m.discrepancies.WithLabelValues(r.Collection, "orphaned").Add(float64(len(r.OrphanedInEngine)))
	m.lastRun.WithLabelValues(r.Collection).Set(r.Duration.Seconds())
}

// ReconcilerConfig wires a Reconciler.
type ReconcilerConfig struct {
	Source SourceScanner
	Index  IndexIDLister
	// Engine receives repair ops when Repair is true.
	Engine EngineClient
	// Repair applies fixes; when false the pass only reports.
	Repair bool
	// PageSize bounds each scan page on both sides. Default 1000.
	PageSize int
	// Metrics, when non-nil, records per-pass observations.
	Metrics *ReconcileMetrics
}

// Reconciler diffs a source table against its engine collection by merging
// two id-ordered streams (SCOPE Task 7.4). Post-quantization vectors are
// lossy, so reconciliation compares id presence, not vector contents.
type Reconciler struct {
	cfg ReconcilerConfig
	tr  *Transformer
}

// NewReconciler creates a Reconciler for the mappings compiled in cfg.
func NewReconciler(syncCfg *SyncConfig, cfg ReconcilerConfig) (*Reconciler, error) {
	if cfg.Source == nil || cfg.Index == nil {
		return nil, fmt.Errorf("replication: reconciler: source and index scanners are required")
	}
	if cfg.Repair && cfg.Engine == nil {
		return nil, fmt.Errorf("replication: reconciler: repair requires an engine client")
	}
	if cfg.PageSize <= 0 {
		cfg.PageSize = 1000
	}
	return &Reconciler{cfg: cfg, tr: NewTransformer(syncCfg)}, nil
}

// sourceCursor pulls filtered, transformed upsert ops from the source in id
// order. Rows failing the filter are skipped (they must not be in the
// engine); malformed rows are counted and skipped.
type sourceCursor struct {
	r       *Reconciler
	mapping TableMapping
	buf     []EngineOp
	afterID string
	hasMore bool
	scanned int
	badRows int
}

func (c *sourceCursor) next(ctx context.Context) (EngineOp, bool, error) {
	for {
		if len(c.buf) > 0 {
			op := c.buf[0]
			c.buf = c.buf[1:]
			return op, true, nil
		}
		if !c.hasMore {
			return EngineOp{}, false, nil
		}
		rows, more, err := c.r.cfg.Source.ScanRows(ctx, c.mapping, c.afterID, c.r.cfg.PageSize)
		if err != nil {
			return EngineOp{}, false, fmt.Errorf("replication: reconcile: scan %s: %w", c.mapping.Postgres, err)
		}
		c.hasMore = more
		for _, row := range rows {
			c.scanned++
			ev := ChangeEvent{Op: OpInsert, Table: c.mapping.Postgres, Row: row}
			op, ok, terr := c.r.tr.Transform(ev)
			if terr != nil {
				c.badRows++
				continue
			}
			if ok {
				c.buf = append(c.buf, op)
			}
		}
		if len(rows) > 0 {
			// Keyset cursor advances past the whole page, even if every
			// row was filtered or malformed.
			c.afterID = rawIDText(rows[len(rows)-1], c.mapping.Columns.ID)
		}
	}
}

// rawIDText renders the id column as scanned (already text from PostgreSQL).
func rawIDText(row map[string]any, col string) string {
	if s, ok := row[col].(string); ok {
		return s
	}
	if id, err := extractID(row, col); err == nil {
		return id
	}
	return ""
}

// engineCursor pulls engine ids in id order.
type engineCursor struct {
	r          *Reconciler
	collection string
	buf        []string
	afterID    string
	hasMore    bool
	scanned    int
}

func (c *engineCursor) next(ctx context.Context) (string, bool, error) {
	for {
		if len(c.buf) > 0 {
			id := c.buf[0]
			c.buf = c.buf[1:]
			c.afterID = id
			c.scanned++
			return id, true, nil
		}
		if !c.hasMore {
			return "", false, nil
		}
		ids, more, err := c.r.cfg.Index.ListIDs(ctx, c.collection, c.afterID, c.r.cfg.PageSize)
		if err != nil {
			return "", false, fmt.Errorf("replication: reconcile: list engine ids %s: %w", c.collection, err)
		}
		c.buf = ids
		c.hasMore = more
		if len(ids) == 0 && !more {
			return "", false, nil
		}
	}
}

// ReconcileTable runs one pass for a single table mapping.
func (r *Reconciler) ReconcileTable(ctx context.Context, mapping TableMapping) (ReconcileReport, error) {
	start := time.Now()
	report := ReconcileReport{Collection: mapping.Engine}

	src := &sourceCursor{r: r, mapping: mapping, hasMore: true}
	eng := &engineCursor{r: r, collection: mapping.Engine, hasMore: true}

	srcOp, srcOK, err := src.next(ctx)
	if err != nil {
		return report, err
	}
	engID, engOK, err := eng.next(ctx)
	if err != nil {
		return report, err
	}

	var repairs []EngineOp
	for srcOK || engOK {
		if ctx.Err() != nil {
			return report, ctx.Err()
		}
		switch {
		case srcOK && engOK && srcOp.ID == engID:
			// In sync: advance both.
			if srcOp, srcOK, err = src.next(ctx); err != nil {
				return report, err
			}
			if engID, engOK, err = eng.next(ctx); err != nil {
				return report, err
			}
		case engOK && (!srcOK || engID < srcOp.ID):
			// Engine has an id the source does not: delete.
			report.OrphanedInEngine = append(report.OrphanedInEngine, engID)
			repairs = append(repairs, EngineOp{Kind: EngineDelete, Collection: mapping.Engine, ID: engID})
			if engID, engOK, err = eng.next(ctx); err != nil {
				return report, err
			}
		default:
			// Source has a row the engine does not: upsert.
			report.MissingInEngine = append(report.MissingInEngine, srcOp.ID)
			repairs = append(repairs, srcOp)
			if srcOp, srcOK, err = src.next(ctx); err != nil {
				return report, err
			}
		}
	}

	report.SourceRows = src.scanned
	report.EngineIDs = eng.scanned
	report.MalformedRows = src.badRows

	if r.cfg.Repair && len(repairs) > 0 {
		w := NewWriter(r.cfg.Engine, WriterConfig{})
		for _, op := range repairs {
			if err := w.Apply(ctx, op); err != nil {
				return report, fmt.Errorf("replication: reconcile: repair %s: %w", op.ID, err)
			}
		}
		if err := w.Flush(ctx); err != nil {
			return report, fmt.Errorf("replication: reconcile: repair flush: %w", err)
		}
		report.Repaired = true
	}

	report.Duration = time.Since(start)
	r.cfg.Metrics.observe(report)
	return report, nil
}
