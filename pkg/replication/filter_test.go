package replication

import (
	"strings"
	"testing"
)

func TestParseFilterErrors(t *testing.T) {
	tests := []struct {
		expr    string
		wantErr string
	}{
		{"", "empty filter"},
		{"   ", "empty filter"},
		{"deleted_at LIKE '%x%'", "unsupported predicate"},
		{"a OR b", "unsupported predicate"},
		{"a > 5", "unsupported predicate"},
		{"a = unquoted", "unsupported literal"},
	}
	for _, tt := range tests {
		t.Run(tt.expr, func(t *testing.T) {
			_, err := ParseFilter(tt.expr)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("error %q does not contain %q", err, tt.wantErr)
			}
		})
	}
}

func TestFilterMatches(t *testing.T) {
	tests := []struct {
		name string
		expr string
		row  map[string]any
		want bool
	}{
		{"is null matches nil", "deleted_at IS NULL", map[string]any{"deleted_at": nil}, true},
		{"is null matches absent", "deleted_at IS NULL", map[string]any{}, true},
		{"is null rejects value", "deleted_at IS NULL", map[string]any{"deleted_at": "2026-01-01"}, false},
		{"is not null", "deleted_at IS NOT NULL", map[string]any{"deleted_at": "x"}, true},
		{"is not null rejects nil", "deleted_at IS NOT NULL", map[string]any{}, false},
		{"case-insensitive keywords", "deleted_at is null", map[string]any{}, true},
		{"string eq", "status = 'active'", map[string]any{"status": "active"}, true},
		{"string eq mismatch", "status = 'active'", map[string]any{"status": "archived"}, false},
		{"string eq null row value", "status = 'active'", map[string]any{}, false},
		{"escaped quote", "note = 'it''s'", map[string]any{"note": "it's"}, true},
		{"neq", "status != 'archived'", map[string]any{"status": "active"}, true},
		{"neq sql form", "status <> 'archived'", map[string]any{"status": "archived"}, false},
		{"neq is false for null", "status != 'archived'", map[string]any{}, false},
		{"number eq float row", "version = 3", map[string]any{"version": 3.0}, true},
		{"number eq int row", "version = 3", map[string]any{"version": int64(3)}, true},
		{"number eq string row", "version = 3", map[string]any{"version": "3"}, true},
		{"bool eq", "published = true", map[string]any{"published": true}, true},
		{"bool eq pgoutput text", "published = true", map[string]any{"published": "t"}, true},
		{"bool eq pgoutput false", "published = false", map[string]any{"published": "f"}, true},
		{"and both hold", "deleted_at IS NULL AND status = 'active'",
			map[string]any{"status": "active"}, true},
		{"and one fails", "deleted_at IS NULL AND status = 'active'",
			map[string]any{"status": "archived"}, false},
		{"and case-insensitive", "deleted_at IS NULL and status = 'active'",
			map[string]any{"status": "active"}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f, err := ParseFilter(tt.expr)
			if err != nil {
				t.Fatalf("ParseFilter(%q): %v", tt.expr, err)
			}
			if got := f.Matches(tt.row); got != tt.want {
				t.Errorf("Matches(%v) = %v, want %v", tt.row, got, tt.want)
			}
		})
	}
}
