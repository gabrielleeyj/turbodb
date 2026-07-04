package replication

import (
	"context"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"
)

// fakeSource serves pre-sorted rows with configurable paging.
type fakeSource struct {
	rows []map[string]any // must be sorted by id text
}

func (f *fakeSource) ScanRows(_ context.Context, mapping TableMapping, afterID string, limit int) ([]map[string]any, bool, error) {
	var out []map[string]any
	for _, r := range f.rows {
		id, _ := r[mapping.Columns.ID].(string)
		if id > afterID {
			out = append(out, r)
		}
		if len(out) == limit+1 {
			break
		}
	}
	hasMore := len(out) > limit
	if hasMore {
		out = out[:limit]
	}
	return out, hasMore, nil
}

// fakeIDLister serves a pre-sorted id list with paging.
type fakeIDLister struct {
	ids []string
}

func (f *fakeIDLister) ListIDs(_ context.Context, _ string, afterID string, pageSize int) ([]string, bool, error) {
	var out []string
	for _, id := range f.ids {
		if id > afterID {
			out = append(out, id)
		}
		if len(out) == pageSize+1 {
			break
		}
	}
	hasMore := len(out) > pageSize
	if hasMore {
		out = out[:pageSize]
	}
	return out, hasMore, nil
}

func docRow(id, vec, deletedAt string) map[string]any {
	row := map[string]any{"doc_id": id, "vector": vec}
	if deletedAt == "" {
		row["deleted_at"] = nil
	} else {
		row["deleted_at"] = deletedAt
	}
	return row
}

func newTestReconciler(t *testing.T, src SourceScanner, idx IndexIDLister, eng EngineClient, repair bool, pageSize int) *Reconciler {
	t.Helper()
	cfg, err := ParseConfig([]byte(validYAML))
	if err != nil {
		t.Fatal(err)
	}
	rec, err := NewReconciler(cfg, ReconcilerConfig{
		Source: src, Index: idx, Engine: eng, Repair: repair, PageSize: pageSize,
	})
	if err != nil {
		t.Fatal(err)
	}
	return rec
}

func docsMapping(t *testing.T) TableMapping {
	t.Helper()
	cfg, err := ParseConfig([]byte(validYAML))
	if err != nil {
		t.Fatal(err)
	}
	return cfg.Tables[0]
}

func TestReconcileInSync(t *testing.T) {
	src := &fakeSource{rows: []map[string]any{
		docRow("a", "[1]", ""), docRow("b", "[2]", ""),
	}}
	idx := &fakeIDLister{ids: []string{"a", "b"}}
	rec := newTestReconciler(t, src, idx, nil, false, 10)

	report, err := rec.ReconcileTable(context.Background(), docsMapping(t))
	if err != nil {
		t.Fatal(err)
	}
	if report.Discrepancies() != 0 {
		t.Errorf("discrepancies: got %d, want 0 (%+v)", report.Discrepancies(), report)
	}
	if report.SourceRows != 2 || report.EngineIDs != 2 {
		t.Errorf("counts: %+v", report)
	}
}

func TestReconcileFindsMissingAndOrphaned(t *testing.T) {
	// Source: a, c (b soft-deleted -> filtered). Engine: b, c, d.
	src := &fakeSource{rows: []map[string]any{
		docRow("a", "[1]", ""), docRow("b", "[2]", "gone"), docRow("c", "[3]", ""),
	}}
	idx := &fakeIDLister{ids: []string{"b", "c", "d"}}
	rec := newTestReconciler(t, src, idx, nil, false, 10)

	report, err := rec.ReconcileTable(context.Background(), docsMapping(t))
	if err != nil {
		t.Fatal(err)
	}
	if len(report.MissingInEngine) != 1 || report.MissingInEngine[0] != "a" {
		t.Errorf("missing: got %v, want [a]", report.MissingInEngine)
	}
	// b is filtered out of the source, so its presence in the engine is an
	// orphan; d simply does not exist in the source.
	if len(report.OrphanedInEngine) != 2 || report.OrphanedInEngine[0] != "b" || report.OrphanedInEngine[1] != "d" {
		t.Errorf("orphaned: got %v, want [b d]", report.OrphanedInEngine)
	}
	if report.Repaired {
		t.Error("repair must not run without --repair")
	}
}

func TestReconcileRepairs(t *testing.T) {
	src := &fakeSource{rows: []map[string]any{docRow("a", "[1]", "")}}
	idx := &fakeIDLister{ids: []string{"z"}}
	eng := &fakeEngine{}
	rec := newTestReconciler(t, src, idx, eng, true, 10)

	report, err := rec.ReconcileTable(context.Background(), docsMapping(t))
	if err != nil {
		t.Fatal(err)
	}
	if !report.Repaired {
		t.Fatal("expected repairs to be applied")
	}
	if len(eng.inserts) != 1 || eng.inserts[0][0].ID != "a" {
		t.Errorf("repair inserts: %v", eng.inserts)
	}
	if len(eng.deletes) != 1 || eng.deletes[0][0] != "z" {
		t.Errorf("repair deletes: %v", eng.deletes)
	}
}

func TestReconcilePaginatesBothSides(t *testing.T) {
	var rows []map[string]any
	var ids []string
	for _, id := range []string{"a", "b", "c", "d", "e", "f", "g"} {
		rows = append(rows, docRow(id, "[1]", ""))
		ids = append(ids, id)
	}
	// Engine is missing "d"; source pages of 2 force multiple scans.
	engIDs := append(append([]string{}, ids[:3]...), ids[4:]...)
	rec := newTestReconciler(t, &fakeSource{rows: rows}, &fakeIDLister{ids: engIDs}, nil, false, 2)

	report, err := rec.ReconcileTable(context.Background(), docsMapping(t))
	if err != nil {
		t.Fatal(err)
	}
	if len(report.MissingInEngine) != 1 || report.MissingInEngine[0] != "d" {
		t.Errorf("missing: got %v, want [d]", report.MissingInEngine)
	}
	if report.SourceRows != 7 || report.EngineIDs != 6 {
		t.Errorf("counts: %+v", report)
	}
}

func TestReconcileCountsMalformedRows(t *testing.T) {
	src := &fakeSource{rows: []map[string]any{
		docRow("a", "[1]", ""),
		{"doc_id": "bad", "vector": "not-a-vector", "deleted_at": nil},
	}}
	idx := &fakeIDLister{ids: []string{"a"}}
	rec := newTestReconciler(t, src, idx, nil, false, 10)

	report, err := rec.ReconcileTable(context.Background(), docsMapping(t))
	if err != nil {
		t.Fatal(err)
	}
	if report.MalformedRows != 1 {
		t.Errorf("malformed: got %d, want 1", report.MalformedRows)
	}
	if report.Discrepancies() != 0 {
		t.Errorf("discrepancies: got %+v", report)
	}
}

func TestReconcileMetricsRecorded(t *testing.T) {
	reg := prometheus.NewRegistry()
	metrics := NewReconcileMetrics(reg)

	cfg, err := ParseConfig([]byte(validYAML))
	if err != nil {
		t.Fatal(err)
	}
	rec, err := NewReconciler(cfg, ReconcilerConfig{
		Source:  &fakeSource{rows: []map[string]any{docRow("a", "[1]", "")}},
		Index:   &fakeIDLister{ids: []string{"z"}},
		Metrics: metrics,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := rec.ReconcileTable(context.Background(), cfg.Tables[0]); err != nil {
		t.Fatal(err)
	}

	families, err := reg.Gather()
	if err != nil {
		t.Fatal(err)
	}
	byName := make(map[string]*dto.MetricFamily)
	for _, f := range families {
		byName[f.GetName()] = f
	}
	disc := byName["turbodb_sync_reconcile_discrepancies_total"]
	if disc == nil {
		t.Fatal("discrepancies metric not registered")
	}
	var total float64
	for _, m := range disc.GetMetric() {
		total += m.GetCounter().GetValue()
	}
	if total != 2 { // 1 missing + 1 orphaned
		t.Errorf("discrepancies total: got %v, want 2", total)
	}
	if byName["turbodb_sync_reconcile_last_run_seconds"] == nil {
		t.Error("last_run metric not registered")
	}
}

func TestNewReconcilerValidation(t *testing.T) {
	cfg, err := ParseConfig([]byte(validYAML))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewReconciler(cfg, ReconcilerConfig{}); err == nil {
		t.Error("expected error without scanners")
	}
	if _, err := NewReconciler(cfg, ReconcilerConfig{
		Source: &fakeSource{}, Index: &fakeIDLister{}, Repair: true,
	}); err == nil {
		t.Error("expected error for repair without engine client")
	}
}
