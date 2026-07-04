package replication

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const validYAML = `
tables:
  - postgres: public.documents
    engine:   docs
    columns:
      id:        doc_id
      embedding: vector
    filter:    "deleted_at IS NULL"
  - postgres: public.images
    engine:   imgs
    columns:
      id:        img_id
      embedding: emb
`

func TestParseConfigValid(t *testing.T) {
	cfg, err := ParseConfig([]byte(validYAML))
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	if len(cfg.Tables) != 2 {
		t.Fatalf("tables: got %d, want 2", len(cfg.Tables))
	}
	docs := cfg.Tables[0]
	if docs.Postgres != "public.documents" || docs.Engine != "docs" {
		t.Errorf("mapping: got %+v", docs)
	}
	if docs.Columns.ID != "doc_id" || docs.Columns.Embedding != "vector" {
		t.Errorf("columns: got %+v", docs.Columns)
	}
	if docs.filter == nil {
		t.Error("filter should be compiled")
	}
	if cfg.Tables[1].filter != nil {
		t.Error("images has no filter; compiled filter should be nil")
	}
}

func TestParseConfigErrors(t *testing.T) {
	tests := []struct {
		name    string
		yaml    string
		wantErr string
	}{
		{"empty tables", "tables: []", "no tables configured"},
		{"missing postgres", `
tables:
  - engine: docs
    columns: {id: a, embedding: b}
`, "postgres table is required"},
		{"unqualified table", `
tables:
  - postgres: documents
    engine: docs
    columns: {id: a, embedding: b}
`, "schema-qualified"},
		{"missing engine", `
tables:
  - postgres: public.documents
    columns: {id: a, embedding: b}
`, "engine collection is required"},
		{"missing id column", `
tables:
  - postgres: public.documents
    engine: docs
    columns: {embedding: b}
`, "columns.id is required"},
		{"missing embedding column", `
tables:
  - postgres: public.documents
    engine: docs
    columns: {id: a}
`, "columns.embedding is required"},
		{"duplicate table", `
tables:
  - postgres: public.documents
    engine: docs
    columns: {id: a, embedding: b}
  - postgres: public.documents
    engine: other
    columns: {id: a, embedding: b}
`, "duplicate postgres table"},
		{"duplicate collection", `
tables:
  - postgres: public.documents
    engine: docs
    columns: {id: a, embedding: b}
  - postgres: public.other
    engine: docs
    columns: {id: a, embedding: b}
`, "duplicate engine collection"},
		{"bad filter", `
tables:
  - postgres: public.documents
    engine: docs
    columns: {id: a, embedding: b}
    filter: "deleted_at LIKE '%x%'"
`, "unsupported predicate"},
		{"unknown yaml field", `
tables:
  - postgres: public.documents
    engine: docs
    columns: {id: a, embedding: b}
    unknown_key: true
`, "unknown_key"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseConfig([]byte(tt.yaml))
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("error %q does not contain %q", err, tt.wantErr)
			}
		})
	}
}

func TestLoadConfigFromFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "sync.yaml")
	if err := os.WriteFile(path, []byte(validYAML), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := LoadConfig(path)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if len(cfg.Tables) != 2 {
		t.Errorf("tables: got %d, want 2", len(cfg.Tables))
	}

	if _, err := LoadConfig(filepath.Join(dir, "missing.yaml")); err == nil {
		t.Error("expected error for missing file")
	}
}
