package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"sort"

	"github.com/gabrielleeyj/turbodb/internal/engine"
	"github.com/gabrielleeyj/turbodb/internal/ioformats"
)

// openEngine boots an engine on dataDir for CLI operations.
func openEngine(dataDir string) (*engine.Engine, error) {
	if dataDir == "" {
		return nil, fmt.Errorf("--data-dir is required")
	}
	return engine.New(engine.Config{DataDir: dataDir})
}

func runImport(args []string) error {
	fs := flag.NewFlagSet("import", flag.ContinueOnError)
	format := fs.String("format", "safetensors", "input format: safetensors|gguf")
	input := fs.String("input", "", "input file path")
	tensor := fs.String("tensor", "", "tensor name (required for gguf)")
	collection := fs.String("collection", "", "destination collection name")
	dataDir := fs.String("data-dir", "", "engine data directory")
	bits := fs.Int("bits", 4, "quantizer bit width (1..8)")
	seed := fs.Uint64("rotator-seed", 0, "rotator seed")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *input == "" || *collection == "" {
		return fmt.Errorf("--input and --collection are required")
	}

	m, err := ioformats.ReadMatrix(ioformats.Format(*format), *input, *tensor)
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "read %d vectors (dim %d) from %s\n", m.Rows, m.Dim, *input)

	eng, err := openEngine(*dataDir)
	if err != nil {
		return err
	}
	defer func() { _ = eng.Close() }()

	ctx := context.Background()
	progress := newProgress(m.Rows)
	n, err := ioformats.ImportMatrix(ctx, eng, m, ioformats.ImportOptions{
		Collection:  *collection,
		BitWidth:    *bits,
		RotatorSeed: *seed,
		Progress:    progress,
	})
	if err != nil {
		return err
	}
	if err := eng.Flush(ctx, *collection); err != nil {
		return fmt.Errorf("flush: %w", err)
	}
	fmt.Fprintf(os.Stderr, "\nimported %d vectors into collection %q\n", n, *collection)
	return nil
}

func runExport(args []string) error {
	fs := flag.NewFlagSet("export", flag.ContinueOnError)
	collection := fs.String("collection", "", "source collection name")
	output := fs.String("output", "", "output SafeTensors file path")
	dataDir := fs.String("data-dir", "", "engine data directory")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *collection == "" || *output == "" {
		return fmt.Errorf("--collection and --output are required")
	}

	eng, err := openEngine(*dataDir)
	if err != nil {
		return err
	}
	defer func() { _ = eng.Close() }()

	cfg, entries, err := eng.ExportCollection(*collection)
	if err != nil {
		return err
	}
	f, err := os.Create(*output)
	if err != nil {
		return fmt.Errorf("create %s: %w", *output, err)
	}
	if err := ioformats.ExportSafeTensors(f, cfg, entries); err != nil {
		_ = f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("close %s: %w", *output, err)
	}
	fmt.Fprintf(os.Stderr, "exported %d vectors from %q to %s\n", len(entries), *collection, *output)
	return nil
}

func runInspect(args []string) error {
	fs := flag.NewFlagSet("inspect", flag.ContinueOnError)
	format := fs.String("format", "safetensors", "format: safetensors|gguf")
	input := fs.String("input", "", "input file path")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *input == "" {
		return fmt.Errorf("--input is required")
	}
	tensors, metadata, err := ioformats.Inspect(ioformats.Format(*format), *input)
	if err != nil {
		return err
	}
	fmt.Printf("%s: %d tensors\n", *input, len(tensors))
	for _, t := range tensors {
		fmt.Printf("  %-40s %-6s %v\n", t.Name, t.Dtype, t.Shape)
	}
	if len(metadata) > 0 {
		fmt.Println("metadata:")
		keys := make([]string, 0, len(metadata))
		for k := range metadata {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			fmt.Printf("  %s = %s\n", k, metadata[k])
		}
	}
	return nil
}

func runConvert(args []string) error {
	fs := flag.NewFlagSet("convert", flag.ContinueOnError)
	from := fs.String("from", "safetensors", "source format (only safetensors supported)")
	to := fs.String("to", "gguf", "destination format (only gguf supported)")
	input := fs.String("input", "", "input file path")
	output := fs.String("output", "", "output file path")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *input == "" || *output == "" {
		return fmt.Errorf("--input and --output are required")
	}
	if *from != "safetensors" || *to != "gguf" {
		return fmt.Errorf("only --from safetensors --to gguf is supported")
	}
	f, err := os.Create(*output)
	if err != nil {
		return fmt.Errorf("create %s: %w", *output, err)
	}
	if err := ioformats.ConvertSafeTensorsToGGUF(*input, f); err != nil {
		_ = f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("close %s: %w", *output, err)
	}
	fmt.Fprintf(os.Stderr, "converted %s -> %s\n", *input, *output)
	return nil
}

// newProgress returns a progress callback that writes a percentage to stderr.
func newProgress(total int) func(done, total int) {
	return func(done, _ int) {
		if total == 0 {
			return
		}
		fmt.Fprintf(os.Stderr, "\rimporting... %d/%d (%d%%)", done, total, done*100/total)
	}
}
