package main

import (
	"context"
	"fmt"

	apiv1 "github.com/gabrielleeyj/turbodb/api/v1"
	"github.com/spf13/cobra"
)

func newCollectionCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "collection",
		Short: "Manage collections on a running engine",
	}
	cmd.AddCommand(
		newCollectionCreateCmd(),
		newCollectionListCmd(),
		newCollectionDescribeCmd(),
		newCollectionDropCmd(),
		newCollectionFlushCmd(),
	)
	return cmd
}

func newCollectionCreateCmd() *cobra.Command {
	var (
		name   string
		dim    int32
		bits   int32
		metric string
		seed   uint64
	)
	cmd := &cobra.Command{
		Use:   "create",
		Short: "Create a collection",
		RunE: func(cmd *cobra.Command, _ []string) error {
			var m apiv1.Metric
			switch metric {
			case "ip", "inner-product":
				m = apiv1.Metric_METRIC_INNER_PRODUCT
			default:
				return fmt.Errorf("unsupported --metric %q (supported: ip)", metric)
			}
			return withEngineClient(cmd, func(ctx context.Context, c apiv1.TurboDBEngineClient) error {
				_, err := c.CreateCollection(ctx, &apiv1.CreateCollectionRequest{
					Config: &apiv1.CollectionConfig{
						Name:        name,
						Dimension:   dim,
						BitWidth:    bits,
						Metric:      m,
						Variant:     apiv1.QuantizationVariant_QUANTIZATION_VARIANT_MSE,
						RotatorSeed: seed,
					},
				})
				if err != nil {
					return err
				}
				fmt.Fprintf(cmd.OutOrStdout(), "collection %q created (dim=%d bits=%d)\n", name, dim, bits)
				return nil
			})
		},
	}
	cmd.Flags().StringVar(&name, "name", "", "collection name (required)")
	cmd.Flags().Int32Var(&dim, "dim", 0, "vector dimension (required)")
	cmd.Flags().Int32Var(&bits, "bits", 4, "quantizer bit width (1..8)")
	cmd.Flags().StringVar(&metric, "metric", "ip", "similarity metric: ip")
	cmd.Flags().Uint64Var(&seed, "rotator-seed", 0, "rotator seed (0 = engine default)")
	_ = cmd.MarkFlagRequired("name")
	_ = cmd.MarkFlagRequired("dim")
	return cmd
}

func newCollectionListCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "list",
		Short: "List collections",
		RunE: func(cmd *cobra.Command, _ []string) error {
			return withEngineClient(cmd, func(ctx context.Context, c apiv1.TurboDBEngineClient) error {
				resp, err := c.ListCollections(ctx, &apiv1.ListCollectionsRequest{})
				if err != nil {
					return err
				}
				if len(resp.GetCollections()) == 0 {
					fmt.Fprintln(cmd.OutOrStdout(), "no collections")
					return nil
				}
				for _, cc := range resp.GetCollections() {
					fmt.Fprintf(cmd.OutOrStdout(), "%-30s dim=%-6d bits=%d\n",
						cc.GetName(), cc.GetDimension(), cc.GetBitWidth())
				}
				return nil
			})
		},
	}
}

func newCollectionDescribeCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "describe <name>",
		Short: "Show a collection's config and stats",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return withEngineClient(cmd, func(ctx context.Context, c apiv1.TurboDBEngineClient) error {
				resp, err := c.DescribeCollection(ctx, &apiv1.DescribeCollectionRequest{Name: args[0]})
				if err != nil {
					return err
				}
				cfg, stats := resp.GetConfig(), resp.GetStats()
				out := cmd.OutOrStdout()
				fmt.Fprintf(out, "name:              %s\n", cfg.GetName())
				fmt.Fprintf(out, "dimension:         %d\n", cfg.GetDimension())
				fmt.Fprintf(out, "bit_width:         %d\n", cfg.GetBitWidth())
				fmt.Fprintf(out, "rotator_seed:      %d\n", cfg.GetRotatorSeed())
				fmt.Fprintf(out, "vector_count:      %d\n", stats.GetVectorCount())
				fmt.Fprintf(out, "sealed_segments:   %d\n", stats.GetSealedSegmentCount())
				fmt.Fprintf(out, "growing_segments:  %d\n", stats.GetGrowingSegmentCount())
				fmt.Fprintf(out, "host_memory_bytes: %d\n", stats.GetHostMemoryBytes())
				return nil
			})
		},
	}
}

func newCollectionDropCmd() *cobra.Command {
	var confirm bool
	cmd := &cobra.Command{
		Use:   "drop <name>",
		Short: "Drop a collection (destructive)",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			if !confirm {
				return fmt.Errorf("dropping %q deletes its in-memory index state; re-run with --confirm", args[0])
			}
			return withEngineClient(cmd, func(ctx context.Context, c apiv1.TurboDBEngineClient) error {
				if _, err := c.DropCollection(ctx, &apiv1.DropCollectionRequest{Name: args[0]}); err != nil {
					return err
				}
				fmt.Fprintf(cmd.OutOrStdout(), "collection %q dropped\n", args[0])
				return nil
			})
		},
	}
	cmd.Flags().BoolVar(&confirm, "confirm", false, "confirm the destructive drop")
	return cmd
}

func newCollectionFlushCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "flush <name>",
		Short: "Seal the collection's growing segment",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return withEngineClient(cmd, func(ctx context.Context, c apiv1.TurboDBEngineClient) error {
				resp, err := c.Flush(ctx, &apiv1.FlushRequest{Collection: args[0]})
				if err != nil {
					return err
				}
				fmt.Fprintf(cmd.OutOrStdout(), "flushed %q: %d segment(s) sealed\n", args[0], resp.GetSegmentsSealed())
				return nil
			})
		},
	}
}
