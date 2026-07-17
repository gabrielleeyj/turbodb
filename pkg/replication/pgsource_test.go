package replication

import (
	"strings"
	"testing"
)

func TestPgSourceConfigValidate(t *testing.T) {
	valid := PgSourceConfig{DSN: "postgres://h/db", Slot: "turbodb_sync", Publication: "turbodb_pub"}

	tests := []struct {
		name    string
		mutate  func(PgSourceConfig) PgSourceConfig
		wantErr string
	}{
		{
			name:   "valid config",
			mutate: func(c PgSourceConfig) PgSourceConfig { return c },
		},
		{
			name:    "missing DSN",
			mutate:  func(c PgSourceConfig) PgSourceConfig { c.DSN = ""; return c },
			wantErr: "DSN is required",
		},
		{
			name:    "missing slot",
			mutate:  func(c PgSourceConfig) PgSourceConfig { c.Slot = ""; return c },
			wantErr: "slot name is required",
		},
		{
			name:    "missing publication",
			mutate:  func(c PgSourceConfig) PgSourceConfig { c.Publication = ""; return c },
			wantErr: "publication name is required",
		},
		{
			name: "publication with single quote rejected",
			mutate: func(c PgSourceConfig) PgSourceConfig {
				// A quote would break out of the pgoutput plugin-argument
				// string literal (publication_names '...').
				c.Publication = "pub', bogus_option '1"
				return c
			},
			wantErr: "invalid publication name",
		},
		{
			name: "publication with space rejected",
			mutate: func(c PgSourceConfig) PgSourceConfig {
				c.Publication = "pub extra"
				return c
			},
			wantErr: "invalid publication name",
		},
		{
			name: "slot with quote rejected",
			mutate: func(c PgSourceConfig) PgSourceConfig {
				c.Slot = `slot"; DROP TABLE x`
				return c
			},
			wantErr: "invalid slot name",
		},
		{
			name: "slot with leading digit rejected",
			mutate: func(c PgSourceConfig) PgSourceConfig {
				c.Slot = "1slot"
				return c
			},
			wantErr: "invalid slot name",
		},
		{
			name: "underscore and digits accepted",
			mutate: func(c PgSourceConfig) PgSourceConfig {
				c.Slot = "_slot_2"
				c.Publication = "pub_v2"
				return c
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			cfg := tt.mutate(valid)

			// Act
			err := cfg.validate()

			// Assert
			if tt.wantErr == "" {
				if err != nil {
					t.Fatalf("validate: %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("error = %v, want substring %q", err, tt.wantErr)
			}
		})
	}
}
