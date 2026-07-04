package main

import (
	"context"
	"fmt"

	apiv1 "github.com/gabrielleeyj/turbodb/api/v1"
	"github.com/gabrielleeyj/turbodb/pkg/replication"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// grpcEngine adapts the engine's gRPC API to replication.EngineClient.
type grpcEngine struct {
	conn   *grpc.ClientConn
	client apiv1.TurboDBEngineClient
}

func dialEngine(addr string) (*grpcEngine, error) {
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, fmt.Errorf("dial engine %s: %w", addr, err)
	}
	return &grpcEngine{conn: conn, client: apiv1.NewTurboDBEngineClient(conn)}, nil
}

func (g *grpcEngine) InsertBatch(ctx context.Context, collection string, records []replication.VectorRecord) error {
	stream, err := g.client.InsertBatch(ctx)
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

func (g *grpcEngine) DeleteBatch(ctx context.Context, collection string, ids []string) error {
	for _, id := range ids {
		if _, err := g.client.Delete(ctx, &apiv1.DeleteRequest{Collection: collection, Id: id}); err != nil {
			return fmt.Errorf("engine delete %s/%s: %w", collection, id, err)
		}
	}
	return nil
}

func (g *grpcEngine) Close() error { return g.conn.Close() }
