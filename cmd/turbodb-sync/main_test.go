package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const validSyncYAML = `tables:
  - postgres: public.docs
    engine: docs
    columns:
      id: doc_id
      embedding: embedding
    filter: "deleted_at IS NULL"
`

func writeConfig(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "sync.yaml")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestRunCommandDispatch(t *testing.T) {
	tests := []struct {
		name    string
		args    []string
		wantErr string
	}{
		{
			name:    "no arguments prints usage",
			args:    nil,
			wantErr: "usage:",
		},
		{
			name:    "unknown subcommand",
			args:    []string{"frobnicate"},
			wantErr: `unknown command "frobnicate"`,
		},
		{
			name:    "run requires DSN",
			args:    []string{"run", "--config", "ignored.yaml"},
			wantErr: "--pg-dsn (or TURBODB_PG_DSN) is required",
		},
		{
			name:    "reconcile requires DSN",
			args:    []string{"reconcile", "--config", "ignored.yaml"},
			wantErr: "--pg-dsn (or TURBODB_PG_DSN) is required",
		},
		{
			name:    "run rejects bad flag",
			args:    []string{"run", "--definitely-not-a-flag"},
			wantErr: "flag provided but not defined",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange: the DSN check must fire from flags alone.
			t.Setenv("TURBODB_PG_DSN", "")

			// Act
			err := run(tt.args)

			// Assert
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("error = %q, want substring %q", err, tt.wantErr)
			}
		})
	}
}

func TestCheckConfig(t *testing.T) {
	tests := []struct {
		name    string
		config  string
		missing bool
		wantErr string
	}{
		{
			name:   "valid config",
			config: validSyncYAML,
		},
		{
			name:    "invalid yaml",
			config:  "tables: [not: [valid",
			wantErr: "parse yaml",
		},
		{
			name:    "no tables",
			config:  "tables: []\n",
			wantErr: "no tables",
		},
		{
			name:    "unknown field rejected",
			config:  "tables:\n  - postgres: p\n    engine: e\n    bogus: x\n",
			wantErr: "parse yaml",
		},
		{
			name:    "missing file",
			missing: true,
			wantErr: "read config",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			var path string
			if tt.missing {
				path = filepath.Join(t.TempDir(), "does-not-exist.yaml")
			} else {
				path = writeConfig(t, tt.config)
			}

			// Act
			err := run([]string{"check-config", "--config", path})

			// Assert
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("check-config: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("error = %v, want substring %q", err, tt.wantErr)
			}
		})
	}
}

func TestRunSyncFailsOnBadConfigBeforeDialing(t *testing.T) {
	// Arrange: a DSN is present but the config is invalid, so the config
	// error must surface without any network activity.
	t.Setenv("TURBODB_PG_DSN", "postgres://invalid")
	path := filepath.Join(t.TempDir(), "missing.yaml")

	// Act
	err := run([]string{"run", "--config", path})

	// Assert
	if err == nil || !strings.Contains(err.Error(), "read config") {
		t.Errorf("error = %v, want config read failure", err)
	}
}

func TestRunSyncFailsOnUnreachablePostgres(t *testing.T) {
	// Arrange: valid config and checkpoint path; the DSN points at a closed
	// loopback port so the replication source dial fails fast.
	t.Setenv("TURBODB_PG_DSN", "")
	config := writeConfig(t, validSyncYAML)
	ckpt := filepath.Join(t.TempDir(), "sync.ckpt")

	// Act
	err := run([]string{"run",
		"--config", config,
		"--checkpoint", ckpt,
		"--engine", "127.0.0.1:1",
		"--pg-dsn", "postgres://u@127.0.0.1:1/db?sslmode=disable&connect_timeout=1",
	})

	// Assert
	if err == nil {
		t.Fatal("expected connection failure")
	}
	if strings.Contains(err.Error(), "required") {
		t.Errorf("error = %q should be a dial failure, not flag validation", err)
	}
}

func TestRunReconcileFailsOnBadConfigBeforeDialing(t *testing.T) {
	t.Setenv("TURBODB_PG_DSN", "postgres://invalid")
	path := filepath.Join(t.TempDir(), "missing.yaml")

	err := run([]string{"reconcile", "--config", path})

	if err == nil || !strings.Contains(err.Error(), "read config") {
		t.Errorf("error = %v, want config read failure", err)
	}
}
