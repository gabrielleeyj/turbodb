package main

import (
	"context"
	"fmt"
	"os"
	"sort"

	"github.com/gabrielleeyj/turbodb/internal/engine"
	"github.com/gabrielleeyj/turbodb/internal/ioformats"
	"github.com/spf13/cobra"
)

// openEngine boots an embedded engine on dataDir for offline file operations.
// The directory must not be in use by a running turbodb-engine.
func openEngine(dataDir string) (*engine.Engine, error) {
	if dataDir == "" {
		return nil, fmt.Errorf("--data-dir is required")
	}
	return engine.New(engine.Config{DataDir: dataDir})
}

func newImportCmd() *cobra.Command {
	var (
		format     string
		input      string
		tensor     string
		collection string
		dataDir    string
		bits       int
		seed       uint64
	)
	cmd := &cobra.Command{
		Use:   "import",
		Short: "Import vectors from a SafeTensors or GGUF file into a collection",
		RunE: func(cmd *cobra.Command, _ []string) error {
			m, err := ioformats.ReadMatrix(ioformats.Format(format), input, tensor)
			if err != nil {
				return err
			}
			fmt.Fprintf(cmd.ErrOrStderr(), "read %d vectors (dim %d) from %s\n", m.Rows, m.Dim, input)

			eng, err := openEngine(dataDir)
			if err != nil {
				return err
			}
			defer func() { _ = eng.Close() }()

			ctx := context.Background()
			n, err := ioformats.ImportMatrix(ctx, eng, m, ioformats.ImportOptions{
				Collection:  collection,
				BitWidth:    bits,
				RotatorSeed: seed,
				Progress:    newProgress(cmd, m.Rows),
			})
			if err != nil {
				return err
			}
			if err := eng.Flush(ctx, collection); err != nil {
				return fmt.Errorf("flush: %w", err)
			}
			fmt.Fprintf(cmd.ErrOrStderr(), "\nimported %d vectors into collection %q\n", n, collection)
			return nil
		},
	}
	cmd.Flags().StringVar(&format, "format", "safetensors", "input format: safetensors|gguf")
	cmd.Flags().StringVar(&input, "input", "", "input file path (required)")
	cmd.Flags().StringVar(&tensor, "tensor", "", "tensor name (required for gguf)")
	cmd.Flags().StringVar(&collection, "collection", "", "destination collection name (required)")
	cmd.Flags().StringVar(&dataDir, "data-dir", "", "engine data directory (required)")
	cmd.Flags().IntVar(&bits, "bits", 4, "quantizer bit width (1..8)")
	cmd.Flags().Uint64Var(&seed, "rotator-seed", 0, "rotator seed")
	_ = cmd.MarkFlagRequired("input")
	_ = cmd.MarkFlagRequired("collection")
	_ = cmd.MarkFlagRequired("data-dir")
	return cmd
}

func newExportCmd() *cobra.Command {
	var (
		collection string
		output     string
		dataDir    string
	)
	cmd := &cobra.Command{
		Use:   "export",
		Short: "Export a collection to a SafeTensors file",
		RunE: func(cmd *cobra.Command, _ []string) error {
			eng, err := openEngine(dataDir)
			if err != nil {
				return err
			}
			defer func() { _ = eng.Close() }()

			cfg, entries, err := eng.ExportCollection(collection)
			if err != nil {
				return err
			}
			f, err := os.Create(output) // #nosec G304 -- operator-supplied output path is the CLI contract
			if err != nil {
				return fmt.Errorf("create %s: %w", output, err)
			}
			if err := ioformats.ExportSafeTensors(f, cfg, entries); err != nil {
				_ = f.Close()
				return err
			}
			if err := f.Close(); err != nil {
				return fmt.Errorf("close %s: %w", output, err)
			}
			fmt.Fprintf(cmd.ErrOrStderr(), "exported %d vectors from %q to %s\n", len(entries), collection, output)
			return nil
		},
	}
	cmd.Flags().StringVar(&collection, "collection", "", "source collection name (required)")
	cmd.Flags().StringVar(&output, "output", "", "output SafeTensors file path (required)")
	cmd.Flags().StringVar(&dataDir, "data-dir", "", "engine data directory (required)")
	_ = cmd.MarkFlagRequired("collection")
	_ = cmd.MarkFlagRequired("output")
	_ = cmd.MarkFlagRequired("data-dir")
	return cmd
}

func newInspectCmd() *cobra.Command {
	var (
		format string
		input  string
	)
	cmd := &cobra.Command{
		Use:   "inspect",
		Short: "List tensors and metadata in a SafeTensors or GGUF file",
		RunE: func(cmd *cobra.Command, _ []string) error {
			tensors, metadata, err := ioformats.Inspect(ioformats.Format(format), input)
			if err != nil {
				return err
			}
			out := cmd.OutOrStdout()
			fmt.Fprintf(out, "%s: %d tensors\n", input, len(tensors))
			for _, t := range tensors {
				fmt.Fprintf(out, "  %-40s %-6s %v\n", t.Name, t.Dtype, t.Shape)
			}
			if len(metadata) > 0 {
				fmt.Fprintln(out, "metadata:")
				keys := make([]string, 0, len(metadata))
				for k := range metadata {
					keys = append(keys, k)
				}
				sort.Strings(keys)
				for _, k := range keys {
					fmt.Fprintf(out, "  %s = %s\n", k, metadata[k])
				}
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&format, "format", "safetensors", "format: safetensors|gguf")
	cmd.Flags().StringVar(&input, "input", "", "input file path (required)")
	_ = cmd.MarkFlagRequired("input")
	return cmd
}

func newConvertCmd() *cobra.Command {
	var (
		from   string
		to     string
		input  string
		output string
	)
	cmd := &cobra.Command{
		Use:   "convert",
		Short: "Convert a SafeTensors file to GGUF",
		RunE: func(cmd *cobra.Command, _ []string) error {
			if from != "safetensors" || to != "gguf" {
				return fmt.Errorf("only --from safetensors --to gguf is supported")
			}
			f, err := os.Create(output) // #nosec G304 -- operator-supplied output path is the CLI contract
			if err != nil {
				return fmt.Errorf("create %s: %w", output, err)
			}
			if err := ioformats.ConvertSafeTensorsToGGUF(input, f); err != nil {
				_ = f.Close()
				return err
			}
			if err := f.Close(); err != nil {
				return fmt.Errorf("close %s: %w", output, err)
			}
			fmt.Fprintf(cmd.ErrOrStderr(), "converted %s -> %s\n", input, output)
			return nil
		},
	}
	cmd.Flags().StringVar(&from, "from", "safetensors", "source format (only safetensors supported)")
	cmd.Flags().StringVar(&to, "to", "gguf", "destination format (only gguf supported)")
	cmd.Flags().StringVar(&input, "input", "", "input file path (required)")
	cmd.Flags().StringVar(&output, "output", "", "output file path (required)")
	_ = cmd.MarkFlagRequired("input")
	_ = cmd.MarkFlagRequired("output")
	return cmd
}

// newProgress returns a progress callback that writes a percentage to the
// command's stderr.
func newProgress(cmd *cobra.Command, total int) func(done, total int) {
	return func(done, _ int) {
		if total == 0 {
			return
		}
		fmt.Fprintf(cmd.ErrOrStderr(), "\rimporting... %d/%d (%d%%)", done, total, done*100/total)
	}
}
