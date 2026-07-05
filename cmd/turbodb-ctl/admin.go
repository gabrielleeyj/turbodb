package main

import (
	"context"
	"fmt"

	apiv1 "github.com/gabrielleeyj/turbodb/api/v1"
	"github.com/gabrielleeyj/turbodb/internal/engine"
	"github.com/spf13/cobra"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func newAdminCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "admin",
		Short: "Administrative operations on a running engine",
	}
	cmd.AddCommand(
		newAdminHealthCmd(),
		newAdminGPUInfoCmd(),
		newAdminRotatorRegenerateCmd(),
		newAdminCodebookUpgradeCmd(),
	)
	return cmd
}

// withAdminClient dials the engine's admin service from --engine.
func withAdminClient(cmd *cobra.Command, fn func(ctx context.Context, c apiv1.TurboDBAdminClient) error) error {
	addr, err := cmd.Flags().GetString("engine")
	if err != nil {
		return err
	}
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return fmt.Errorf("dial engine %s: %w", addr, err)
	}
	defer func() { _ = conn.Close() }()

	ctx, cancel := context.WithTimeout(cmd.Context(), rpcTimeout)
	defer cancel()
	return fn(ctx, apiv1.NewTurboDBAdminClient(conn))
}

func newAdminHealthCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "health",
		Short: "Show engine health, version, and uptime",
		RunE: func(cmd *cobra.Command, _ []string) error {
			return withAdminClient(cmd, func(ctx context.Context, c apiv1.TurboDBAdminClient) error {
				h, err := c.Health(ctx, &apiv1.HealthRequest{})
				if err != nil {
					return err
				}
				r, err := c.Ready(ctx, &apiv1.ReadyRequest{})
				if err != nil {
					return err
				}
				out := cmd.OutOrStdout()
				fmt.Fprintf(out, "healthy: %v\n", h.GetHealthy())
				fmt.Fprintf(out, "ready:   %v\n", r.GetReady())
				fmt.Fprintf(out, "version: %s\n", h.GetVersion())
				fmt.Fprintf(out, "uptime:  %s\n", h.GetUptime())
				return nil
			})
		},
	}
}

func newAdminGPUInfoCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "gpu-info",
		Short: "Show CUDA devices visible to the engine",
		RunE: func(cmd *cobra.Command, _ []string) error {
			return withAdminClient(cmd, func(ctx context.Context, c apiv1.TurboDBAdminClient) error {
				resp, err := c.GPUInfo(ctx, &apiv1.GPUInfoRequest{})
				if err != nil {
					return err
				}
				out := cmd.OutOrStdout()
				if len(resp.GetDevices()) == 0 {
					fmt.Fprintln(out, "no GPU devices (engine built without CUDA support or no device present)")
					return nil
				}
				for _, d := range resp.GetDevices() {
					fmt.Fprintf(out, "device %d: compute=%s total=%d free=%d\n",
						d.GetId(), d.GetComputeCapability(),
						d.GetTotalMemoryBytes(), d.GetFreeMemoryBytes())
				}
				return nil
			})
		},
	}
}

func newAdminRotatorRegenerateCmd() *cobra.Command {
	var confirm string
	cmd := &cobra.Command{
		Use:   "rotator-regenerate <collection>",
		Short: "Regenerate a collection's rotator seed (SECURITY SENSITIVE, invalidates all indexes)",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			if confirm != engine.RotatorRegenerateConfirmPhrase {
				return fmt.Errorf("pass --confirm %q to proceed", engine.RotatorRegenerateConfirmPhrase)
			}
			return withAdminClient(cmd, func(ctx context.Context, c apiv1.TurboDBAdminClient) error {
				resp, err := c.RotatorRegenerate(ctx, &apiv1.RotatorRegenerateRequest{
					Collection: args[0],
					Confirm:    confirm,
				})
				if err != nil {
					return err
				}
				fmt.Fprintln(cmd.OutOrStdout(), resp.GetMessage())
				return nil
			})
		},
	}
	cmd.Flags().StringVar(&confirm, "confirm", "", "must be exactly: "+engine.RotatorRegenerateConfirmPhrase)
	return cmd
}

func newAdminCodebookUpgradeCmd() *cobra.Command {
	var from, to string
	cmd := &cobra.Command{
		Use:   "codebook-upgrade <collection>",
		Short: "Upgrade a collection's codebook version",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return withAdminClient(cmd, func(ctx context.Context, c apiv1.TurboDBAdminClient) error {
				resp, err := c.CodebookUpgrade(ctx, &apiv1.CodebookUpgradeRequest{
					Collection:  args[0],
					FromVersion: from,
					ToVersion:   to,
				})
				if err != nil {
					return err
				}
				fmt.Fprintln(cmd.OutOrStdout(), resp.GetMessage())
				return nil
			})
		},
	}
	cmd.Flags().StringVar(&from, "from", "v1", "current codebook version")
	cmd.Flags().StringVar(&to, "to", "", "target codebook version")
	return cmd
}
