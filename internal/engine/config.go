// Package engine implements the standalone vector engine for TurboDB.
// It composes pkg/index, pkg/wal, and the quantization stack into a
// crash-recoverable, gRPC-fronted service.
package engine

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
)

// Variant identifies the quantization algorithm.
type Variant string

const (
	// VariantMSE selects MSEQuantizer (Algorithm 1). MVP-supported.
	VariantMSE Variant = "mse"
	// VariantProd selects ProdQuantizer (Algorithm 2 with QJL sketch).
	// Not yet supported by the engine; reserved for future.
	VariantProd Variant = "prod"
)

// Metric identifies the similarity metric.
type Metric string

const (
	// MetricInnerProduct is the only metric supported by MVP.
	MetricInnerProduct Metric = "inner_product"
)

// CollectionConfig is the persistent description of a collection.
type CollectionConfig struct {
	Name        string  `json:"name"`
	Dim         int     `json:"dim"`
	BitWidth    int     `json:"bit_width"`
	Metric      Metric  `json:"metric"`
	Variant     Variant `json:"variant"`
	RotatorSeed uint64  `json:"rotator_seed"`
}

// Validate enforces collection-level constraints.
func (c CollectionConfig) Validate() error {
	if c.Name == "" {
		return fmt.Errorf("collection: name must not be empty")
	}
	if len(c.Name) > 128 {
		return fmt.Errorf("collection: name exceeds 128 chars")
	}
	if c.Dim < 1 || c.Dim > 8192 {
		return fmt.Errorf("collection: dim must be 1..8192, got %d", c.Dim)
	}
	if c.BitWidth < 1 || c.BitWidth > 8 {
		return fmt.Errorf("collection: bit_width must be 1..8, got %d", c.BitWidth)
	}
	if c.Metric != MetricInnerProduct {
		return fmt.Errorf("collection: metric %q not supported (only %q)", c.Metric, MetricInnerProduct)
	}
	if c.Variant != VariantMSE {
		return fmt.Errorf("collection: variant %q not supported (only %q)", c.Variant, VariantMSE)
	}
	return nil
}

// EngineConfig configures the Engine.
type EngineConfig struct {
	// DataDir holds collection metadata, sealed segments, and the WAL.
	DataDir string
	// SealThreshold is the number of vectors per growing segment before auto-seal.
	// Defaults to index.DefaultSealThreshold (1M).
	SealThreshold int
	// Logger receives operational logs. Defaults to slog.Default().
	Logger *slog.Logger
}

// Validate ensures the engine config has the minimum required fields.
func (c EngineConfig) Validate() error {
	if c.DataDir == "" {
		return fmt.Errorf("engine: DataDir must not be empty")
	}
	return nil
}

// configsDir returns the directory where collection configs are persisted.
func configsDir(dataDir string) string {
	return filepath.Join(dataDir, "collections")
}

// walDir returns the directory where WAL files are persisted.
func walDir(dataDir string) string {
	return filepath.Join(dataDir, "wal")
}

// segmentsDir returns the directory where sealed segment files are persisted.
func segmentsDir(dataDir string) string {
	return filepath.Join(dataDir, "segments")
}

// configPath returns the JSON file path for a collection's config.
func configPath(dataDir, name string) string {
	return filepath.Join(configsDir(dataDir), name+".json")
}

// saveCollectionConfig writes a collection config to disk atomically.
func saveCollectionConfig(dataDir string, cfg CollectionConfig) error {
	if err := os.MkdirAll(configsDir(dataDir), 0o755); err != nil {
		return fmt.Errorf("engine: create configs dir: %w", err)
	}
	path := configPath(dataDir, cfg.Name)
	tmp := path + ".tmp"

	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return fmt.Errorf("engine: marshal config: %w", err)
	}
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("engine: write tmp config: %w", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		return fmt.Errorf("engine: rename config: %w", err)
	}
	return nil
}

// removeCollectionConfig deletes a collection's config from disk.
func removeCollectionConfig(dataDir, name string) error {
	path := configPath(dataDir, name)
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("engine: remove config: %w", err)
	}
	return nil
}

// loadCollectionConfigs reads all collection configs from disk.
func loadCollectionConfigs(dataDir string) ([]CollectionConfig, error) {
	dir := configsDir(dataDir)
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("engine: read configs dir: %w", err)
	}

	var configs []CollectionConfig
	for _, e := range entries {
		if e.IsDir() || filepath.Ext(e.Name()) != ".json" {
			continue
		}
		path := filepath.Join(dir, e.Name())
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("engine: read %s: %w", path, err)
		}
		var cfg CollectionConfig
		if err := json.Unmarshal(data, &cfg); err != nil {
			return nil, fmt.Errorf("engine: parse %s: %w", path, err)
		}
		if err := cfg.Validate(); err != nil {
			return nil, fmt.Errorf("engine: invalid config %s: %w", path, err)
		}
		configs = append(configs, cfg)
	}
	return configs, nil
}
