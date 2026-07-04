package main

import (
	"context"
	"fmt"
	"time"

	apiv1 "github.com/gabrielleeyj/turbodb/api/v1"
	"github.com/spf13/cobra"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// rpcTimeout bounds every single CLI-issued RPC.
const rpcTimeout = 30 * time.Second

// withEngineClient dials the engine from the command's --engine flag, runs
// fn with a bounded context, and closes the connection.
func withEngineClient(cmd *cobra.Command, fn func(ctx context.Context, c apiv1.TurboDBEngineClient) error) error {
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
	return fn(ctx, apiv1.NewTurboDBEngineClient(conn))
}
