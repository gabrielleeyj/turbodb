package main

import (
	"context"
	"fmt"

	apiv1 "github.com/gabrielleeyj/turbodb/api/v1"
	"github.com/spf13/cobra"
)

func newIndexCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "index",
		Short: "Index maintenance and statistics",
	}
	cmd.AddCommand(newIndexBuildStatsCmd())
	// "index compact" (SCOPE Task 8.1) is pending engine-side segment
	// compaction support and is intentionally not registered yet.
	return cmd
}

func newIndexBuildStatsCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "build-stats <collection>",
		Short: "Show index build statistics for a collection",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return withEngineClient(cmd, func(ctx context.Context, c apiv1.TurboDBEngineClient) error {
				resp, err := c.GetStats(ctx, &apiv1.GetStatsRequest{Collection: args[0]})
				if err != nil {
					return err
				}
				s := resp.GetStats()
				out := cmd.OutOrStdout()
				fmt.Fprintf(out, "vector_count:      %d\n", s.GetVectorCount())
				fmt.Fprintf(out, "sealed_segments:   %d\n", s.GetSealedSegmentCount())
				fmt.Fprintf(out, "growing_segments:  %d\n", s.GetGrowingSegmentCount())
				fmt.Fprintf(out, "host_memory_bytes: %d\n", s.GetHostMemoryBytes())
				fmt.Fprintf(out, "gpu_memory_bytes:  %d\n", s.GetGpuMemoryBytes())
				return nil
			})
		},
	}
}
