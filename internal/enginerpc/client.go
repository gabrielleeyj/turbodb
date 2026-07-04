// Package enginerpc adapts the engine's gRPC API to the interfaces the
// replication pipeline consumes (EngineClient, IndexIDLister). Shared by
// turbodb-sync and turbodb-ctl.
package enginerpc

import (
	"context"
	"fmt"

	apiv1 "github.com/gabrielleeyj/turbodb/api/v1"
	"github.com/gabrielleeyj/turbodb/pkg/replication"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// Client is a gRPC-backed replication.EngineClient and
// replication.IndexIDLister.
type Client struct {
	conn   *grpc.ClientConn
	client apiv1.TurboDBEngineClient
}

// Dial connects to a turbodb-engine gRPC address.
func Dial(addr string) (*Client, error) {
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("dial engine %s: %w", addr, err)
	}
	return &Client{conn: conn, client: apiv1.NewTurboDBEngineClient(conn)}, nil
}

// InsertBatch streams one batch of vectors into a collection.
func (c *Client) InsertBatch(ctx context.Context, collection string, records []replication.VectorRecord) error {
	stream, err := c.client.InsertBatch(ctx)
	if err != nil {
		return fmt.Errorf("engine insert batch: %w", err)
	}
	vectors := make([]*apiv1.Vector, len(records))
	for i, r := range records {
		vectors[i] = &apiv1.Vector{Id: r.ID, Values: r.Vector}
	}
	if err := stream.Send(&apiv1.InsertBatchRequest{Collection: collection, Vectors: vectors}); err != nil {
		return fmt.Errorf("engine insert batch send: %w", err)
	}
	if _, err := stream.CloseAndRecv(); err != nil {
		return fmt.Errorf("engine insert batch close: %w", err)
	}
	return nil
}

// DeleteBatch deletes ids from a collection.
func (c *Client) DeleteBatch(ctx context.Context, collection string, ids []string) error {
	for _, id := range ids {
		if _, err := c.client.Delete(ctx, &apiv1.DeleteRequest{Collection: collection, Id: id}); err != nil {
			return fmt.Errorf("engine delete %s/%s: %w", collection, id, err)
		}
	}
	return nil
}

// ListIDs pages a collection's live vector ids via the engine's ListIDs RPC.
func (c *Client) ListIDs(ctx context.Context, collection, afterID string, pageSize int) ([]string, bool, error) {
	resp, err := c.client.ListIDs(ctx, &apiv1.ListIDsRequest{
		Collection: collection,
		AfterId:    afterID,
		PageSize:   int32(pageSize), // #nosec G115 -- page sizes are small
	})
	if err != nil {
		return nil, false, fmt.Errorf("engine list ids: %w", err)
	}
	return resp.GetIds(), resp.GetHasMore(), nil
}

// Close releases the connection.
func (c *Client) Close() error { return c.conn.Close() }
