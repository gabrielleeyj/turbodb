package replication

import (
	"fmt"
	"os"
	"strings"

	"gopkg.in/yaml.v3"
)

// ColumnMapping names the PostgreSQL columns that feed the engine collection.
type ColumnMapping struct {
	// ID is the source column used as the collection primary key.
	ID string `yaml:"id"`
	// Embedding is the source column holding the vector.
	Embedding string `yaml:"embedding"`
}

// TableMapping maps one PostgreSQL table to one engine collection.
type TableMapping struct {
	// Postgres is the schema-qualified source table ("public.documents").
	Postgres string `yaml:"postgres"`
	// Engine is the destination collection name.
	Engine string `yaml:"engine"`
	// Columns maps source columns to their roles.
	Columns ColumnMapping `yaml:"columns"`
	// Filter is an optional row predicate; rows that do not match are not
	// replicated. See ParseFilter for the supported subset.
	Filter string `yaml:"filter"`

	// filter is the compiled form of Filter, set by validate.
	filter *Filter
}

// SyncConfig is the parsed sync.yaml.
type SyncConfig struct {
	Tables []TableMapping `yaml:"tables"`
}

// LoadConfig reads and validates a sync.yaml file.
func LoadConfig(path string) (*SyncConfig, error) {
	data, err := os.ReadFile(path) // #nosec G304 -- config path is supplied by the operator
	if err != nil {
		return nil, fmt.Errorf("replication: read config %s: %w", path, err)
	}
	cfg, err := ParseConfig(data)
	if err != nil {
		return nil, fmt.Errorf("replication: config %s: %w", path, err)
	}
	return cfg, nil
}

// ParseConfig parses and validates sync.yaml contents.
func ParseConfig(data []byte) (*SyncConfig, error) {
	var cfg SyncConfig
	dec := yaml.NewDecoder(strings.NewReader(string(data)))
	dec.KnownFields(true)
	if err := dec.Decode(&cfg); err != nil {
		return nil, fmt.Errorf("parse yaml: %w", err)
	}
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	return &cfg, nil
}

func (c *SyncConfig) validate() error {
	if len(c.Tables) == 0 {
		return fmt.Errorf("no tables configured")
	}
	seenTable := make(map[string]bool, len(c.Tables))
	seenEngine := make(map[string]bool, len(c.Tables))
	for i := range c.Tables {
		t := &c.Tables[i]
		if err := validateMapping(t); err != nil {
			return fmt.Errorf("tables[%d]: %w", i, err)
		}
		if seenTable[t.Postgres] {
			return fmt.Errorf("tables[%d]: duplicate postgres table %q", i, t.Postgres)
		}
		if seenEngine[t.Engine] {
			return fmt.Errorf("tables[%d]: duplicate engine collection %q", i, t.Engine)
		}
		seenTable[t.Postgres] = true
		seenEngine[t.Engine] = true

		if t.Filter != "" {
			f, err := ParseFilter(t.Filter)
			if err != nil {
				return fmt.Errorf("tables[%d]: filter: %w", i, err)
			}
			t.filter = f
		}
	}
	return nil
}

func validateMapping(t *TableMapping) error {
	if t.Postgres == "" {
		return fmt.Errorf("postgres table is required")
	}
	if !strings.Contains(t.Postgres, ".") {
		return fmt.Errorf("postgres table %q must be schema-qualified (e.g. public.documents)", t.Postgres)
	}
	if t.Engine == "" {
		return fmt.Errorf("engine collection is required")
	}
	if t.Columns.ID == "" {
		return fmt.Errorf("columns.id is required")
	}
	if t.Columns.Embedding == "" {
		return fmt.Errorf("columns.embedding is required")
	}
	return nil
}

// mappingFor returns the table mapping for a schema-qualified table name.
func (c *SyncConfig) mappingFor(table string) (*TableMapping, bool) {
	for i := range c.Tables {
		if c.Tables[i].Postgres == table {
			return &c.Tables[i], true
		}
	}
	return nil, false
}
