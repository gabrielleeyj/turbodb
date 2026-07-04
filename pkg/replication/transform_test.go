package replication

import (
	"strings"
	"testing"
)

func testTransformer(t *testing.T) *Transformer {
	t.Helper()
	cfg, err := ParseConfig([]byte(validYAML))
	if err != nil {
		t.Fatal(err)
	}
	return NewTransformer(cfg)
}

func TestTransformInsert(t *testing.T) {
	tr := testTransformer(t)

	op, ok, err := tr.Transform(ChangeEvent{
		Op: OpInsert, Table: "public.documents", LSN: 42,
		Row: map[string]any{"doc_id": "d1", "vector": "[0.5, -1.25, 2]"},
	})
	if err != nil || !ok {
		t.Fatalf("Transform: ok=%v err=%v", ok, err)
	}
	if op.Kind != EngineUpsert || op.Collection != "docs" || op.ID != "d1" || op.LSN != 42 {
		t.Errorf("op: %+v", op)
	}
	want := []float32{0.5, -1.25, 2}
	if len(op.Vector) != len(want) {
		t.Fatalf("vector len: got %d, want %d", len(op.Vector), len(want))
	}
	for i := range want {
		if op.Vector[i] != want[i] {
			t.Errorf("vector[%d]: got %v, want %v", i, op.Vector[i], want[i])
		}
	}
}

func TestTransformSkipsAndDeletes(t *testing.T) {
	tr := testTransformer(t)
	tests := []struct {
		name     string
		ev       ChangeEvent
		wantOK   bool
		wantKind EngineOpKind
	}{
		{"unconfigured table skipped",
			ChangeEvent{Op: OpInsert, Table: "public.other",
				Row: map[string]any{"doc_id": "x", "vector": "[1]"}},
			false, 0},
		{"insert failing filter skipped",
			ChangeEvent{Op: OpInsert, Table: "public.documents",
				Row: map[string]any{"doc_id": "x", "vector": "[1]", "deleted_at": "2026-01-01"}},
			false, 0},
		{"update failing filter becomes delete",
			ChangeEvent{Op: OpUpdate, Table: "public.documents", LSN: 7,
				Row: map[string]any{"doc_id": "x", "vector": "[1]", "deleted_at": "2026-01-01"}},
			true, EngineDelete},
		{"update passing filter upserts",
			ChangeEvent{Op: OpUpdate, Table: "public.documents",
				Row: map[string]any{"doc_id": "x", "vector": "[1]"}},
			true, EngineUpsert},
		{"delete maps to delete",
			ChangeEvent{Op: OpDelete, Table: "public.documents",
				Row: map[string]any{"doc_id": "x"}},
			true, EngineDelete},
		{"no filter table passes without filter columns",
			ChangeEvent{Op: OpInsert, Table: "public.images",
				Row: map[string]any{"img_id": int64(9), "emb": "{1,2}"}},
			true, EngineUpsert},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op, ok, err := tr.Transform(tt.ev)
			if err != nil {
				t.Fatalf("Transform: %v", err)
			}
			if ok != tt.wantOK {
				t.Fatalf("ok: got %v, want %v", ok, tt.wantOK)
			}
			if ok && op.Kind != tt.wantKind {
				t.Errorf("kind: got %d, want %d", op.Kind, tt.wantKind)
			}
		})
	}
}

func TestTransformIDCoercion(t *testing.T) {
	tr := testTransformer(t)
	tests := []struct {
		name   string
		id     any
		wantID string
	}{
		{"string", "d1", "d1"},
		{"int64", int64(42), "42"},
		{"int", 7, "7"},
		{"float64 integral", float64(3), "3"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			op, ok, err := tr.Transform(ChangeEvent{
				Op: OpDelete, Table: "public.documents",
				Row: map[string]any{"doc_id": tt.id},
			})
			if err != nil || !ok {
				t.Fatalf("ok=%v err=%v", ok, err)
			}
			if op.ID != tt.wantID {
				t.Errorf("id: got %q, want %q", op.ID, tt.wantID)
			}
		})
	}
}

func TestTransformErrors(t *testing.T) {
	tr := testTransformer(t)
	tests := []struct {
		name    string
		row     map[string]any
		wantErr string
	}{
		{"missing id", map[string]any{"vector": "[1]"}, "id column"},
		{"empty id", map[string]any{"doc_id": "", "vector": "[1]"}, "is empty"},
		{"missing vector", map[string]any{"doc_id": "d"}, "embedding column"},
		{"malformed vector", map[string]any{"doc_id": "d", "vector": "not-a-vec"}, "malformed vector"},
		{"empty vector", map[string]any{"doc_id": "d", "vector": "[]"}, "empty vector"},
		{"bad element", map[string]any{"doc_id": "d", "vector": "[1,x]"}, "element 1"},
		{"unsupported id type", map[string]any{"doc_id": []byte("x"), "vector": "[1]"}, "unsupported type"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, err := tr.Transform(ChangeEvent{Op: OpInsert, Table: "public.documents", Row: tt.row})
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("error %q does not contain %q", err, tt.wantErr)
			}
		})
	}
}

func TestExtractVectorForms(t *testing.T) {
	row := func(v any) map[string]any { return map[string]any{"emb": v} }
	tests := []struct {
		name string
		val  any
		want int
	}{
		{"pgvector text", "[1,2,3]", 3},
		{"array text", "{1, 2}", 2},
		{"float32 slice", []float32{1, 2, 3, 4}, 4},
		{"float64 slice", []float64{1, 2}, 2},
		{"any slice", []any{1.0, 2.0, 3.0}, 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			vec, err := extractVector(row(tt.val), "emb")
			if err != nil {
				t.Fatalf("extractVector: %v", err)
			}
			if len(vec) != tt.want {
				t.Errorf("len: got %d, want %d", len(vec), tt.want)
			}
		})
	}
}
