package engine

import (
	"context"
	"errors"
	"fmt"
	"io"
	"time"

	apiv1 "github.com/gabrielleeyj/turbodb/api/v1"
	"github.com/gabrielleeyj/turbodb/pkg/index"
	"github.com/gabrielleeyj/turbodb/pkg/search"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// GRPCServer adapts an Engine to the apiv1.TurboDBEngineServer interface.
type GRPCServer struct {
	apiv1.UnimplementedTurboDBEngineServer
	engine *Engine
}

// NewGRPCServer wraps an Engine for gRPC dispatch.
func NewGRPCServer(e *Engine) *GRPCServer {
	return &GRPCServer{engine: e}
}

// CreateCollection registers a new collection.
func (s *GRPCServer) CreateCollection(ctx context.Context, req *apiv1.CreateCollectionRequest) (*apiv1.CreateCollectionResponse, error) {
	if req.GetConfig() == nil {
		return nil, status.Error(codes.InvalidArgument, "config is required")
	}
	cfg, err := protoToCollectionConfig(req.GetConfig())
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "%v", err)
	}
	if err := s.engine.CreateCollection(ctx, cfg); err != nil {
		return nil, mapError(err)
	}
	return &apiv1.CreateCollectionResponse{Name: cfg.Name}, nil
}

// DropCollection removes a collection.
func (s *GRPCServer) DropCollection(ctx context.Context, req *apiv1.DropCollectionRequest) (*apiv1.DropCollectionResponse, error) {
	if req.GetName() == "" {
		return nil, status.Error(codes.InvalidArgument, "name is required")
	}
	if err := s.engine.DropCollection(ctx, req.GetName()); err != nil {
		return nil, mapError(err)
	}
	return &apiv1.DropCollectionResponse{}, nil
}

// ListCollections returns all collection configs.
func (s *GRPCServer) ListCollections(ctx context.Context, req *apiv1.ListCollectionsRequest) (*apiv1.ListCollectionsResponse, error) {
	configs := s.engine.ListCollections()
	out := make([]*apiv1.CollectionConfig, 0, len(configs))
	for _, c := range configs {
		out = append(out, collectionConfigToProto(c))
	}
	return &apiv1.ListCollectionsResponse{Collections: out}, nil
}

// DescribeCollection returns config + stats for a collection.
func (s *GRPCServer) DescribeCollection(ctx context.Context, req *apiv1.DescribeCollectionRequest) (*apiv1.DescribeCollectionResponse, error) {
	cfg, stats, err := s.engine.DescribeCollection(req.GetName())
	if err != nil {
		return nil, mapError(err)
	}
	return &apiv1.DescribeCollectionResponse{
		Config: collectionConfigToProto(cfg),
		Stats:  collectionStatsToProto(stats),
	}, nil
}

// Insert appends a single vector.
func (s *GRPCServer) Insert(ctx context.Context, req *apiv1.InsertRequest) (*apiv1.InsertResponse, error) {
	if req.GetCollection() == "" {
		return nil, status.Error(codes.InvalidArgument, "collection is required")
	}
	v := req.GetVector()
	if v == nil {
		return nil, status.Error(codes.InvalidArgument, "vector is required")
	}
	entry := index.VectorEntry{ID: v.GetId(), Values: v.GetValues(), Metadata: v.GetMetadata()}
	if err := s.engine.Insert(ctx, req.GetCollection(), entry); err != nil {
		return nil, mapError(err)
	}
	return &apiv1.InsertResponse{Id: v.GetId()}, nil
}

// InsertBatch streams a batch of vector inserts. The server consumes all
// stream messages and returns a single response with the count.
func (s *GRPCServer) InsertBatch(stream apiv1.TurboDBEngine_InsertBatchServer) error {
	var inserted int64
	ctx := stream.Context()
	for {
		req, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return mapError(err)
		}
		coll := req.GetCollection()
		if coll == "" {
			return status.Error(codes.InvalidArgument, "collection is required")
		}
		for _, v := range req.GetVectors() {
			entry := index.VectorEntry{ID: v.GetId(), Values: v.GetValues(), Metadata: v.GetMetadata()}
			if err := s.engine.Insert(ctx, coll, entry); err != nil {
				return mapError(err)
			}
			inserted++
		}
	}
	return stream.SendAndClose(&apiv1.InsertBatchResponse{InsertedCount: inserted})
}

// Delete tombstones a vector.
func (s *GRPCServer) Delete(ctx context.Context, req *apiv1.DeleteRequest) (*apiv1.DeleteResponse, error) {
	if req.GetCollection() == "" {
		return nil, status.Error(codes.InvalidArgument, "collection is required")
	}
	if req.GetId() == "" {
		return nil, status.Error(codes.InvalidArgument, "id is required")
	}
	if err := s.engine.Delete(ctx, req.GetCollection(), req.GetId()); err != nil {
		return nil, mapError(err)
	}
	return &apiv1.DeleteResponse{}, nil
}

// Search runs a similarity query.
func (s *GRPCServer) Search(ctx context.Context, req *apiv1.SearchRequest) (*apiv1.SearchResponse, error) {
	start := time.Now()
	opts := searchOptionsFromProto(req)
	results, _, err := s.engine.Search(ctx, req.GetCollection(), req.GetQuery(), opts)
	if err != nil {
		return nil, mapError(err)
	}
	return &apiv1.SearchResponse{
		Results:      searchResultsToProto(results),
		SearchTimeUs: time.Since(start).Microseconds(),
	}, nil
}

// SearchBatch runs many queries and streams results in order.
func (s *GRPCServer) SearchBatch(req *apiv1.SearchBatchRequest, stream apiv1.TurboDBEngine_SearchBatchServer) error {
	ctx := stream.Context()
	for _, q := range req.GetQueries() {
		start := time.Now()
		opts := searchOptionsFromProto(q)
		results, _, err := s.engine.Search(ctx, q.GetCollection(), q.GetQuery(), opts)
		if err != nil {
			return mapError(err)
		}
		resp := &apiv1.SearchResponse{
			Results:      searchResultsToProto(results),
			SearchTimeUs: time.Since(start).Microseconds(),
		}
		if err := stream.Send(&apiv1.SearchBatchResponse{Response: resp}); err != nil {
			return mapError(err)
		}
	}
	return nil
}

// searchOptionsFromProto translates the proto SearchRequest knobs into the
// planner's Options. Validation is left to the planner.
func searchOptionsFromProto(req *apiv1.SearchRequest) search.Options {
	return search.Options{
		TopK:     int(req.GetTopK()),
		Rerank:   req.GetRerank(),
		Exact:    req.GetExact(),
		EfSearch: int(req.GetEfSearch()),
	}
}

// Flush forces a segment seal in a collection.
func (s *GRPCServer) Flush(ctx context.Context, req *apiv1.FlushRequest) (*apiv1.FlushResponse, error) {
	if req.GetCollection() == "" {
		return nil, status.Error(codes.InvalidArgument, "collection is required")
	}
	if err := s.engine.Flush(ctx, req.GetCollection()); err != nil {
		return nil, mapError(err)
	}
	return &apiv1.FlushResponse{SegmentsSealed: 1}, nil
}

// GetStats returns runtime stats for a collection.
func (s *GRPCServer) GetStats(ctx context.Context, req *apiv1.GetStatsRequest) (*apiv1.GetStatsResponse, error) {
	stats, err := s.engine.Stats(req.GetCollection())
	if err != nil {
		return nil, mapError(err)
	}
	return &apiv1.GetStatsResponse{Stats: collectionStatsToProto(stats)}, nil
}

// --- mappers ---

func protoToCollectionConfig(p *apiv1.CollectionConfig) (CollectionConfig, error) {
	cfg := CollectionConfig{
		Name:        p.GetName(),
		Dim:         int(p.GetDimension()),
		BitWidth:    int(p.GetBitWidth()),
		RotatorSeed: p.GetRotatorSeed(),
	}
	switch p.GetMetric() {
	case apiv1.Metric_METRIC_INNER_PRODUCT:
		cfg.Metric = MetricInnerProduct
	case apiv1.Metric_METRIC_UNSPECIFIED:
		return cfg, fmt.Errorf("metric is required")
	default:
		return cfg, fmt.Errorf("metric %v not supported", p.GetMetric())
	}
	switch p.GetVariant() {
	case apiv1.QuantizationVariant_QUANTIZATION_VARIANT_MSE:
		cfg.Variant = VariantMSE
	case apiv1.QuantizationVariant_QUANTIZATION_VARIANT_UNSPECIFIED:
		return cfg, fmt.Errorf("variant is required")
	default:
		return cfg, fmt.Errorf("variant %v not supported", p.GetVariant())
	}
	return cfg, cfg.Validate()
}

func collectionConfigToProto(c CollectionConfig) *apiv1.CollectionConfig {
	pc := &apiv1.CollectionConfig{
		Name:        c.Name,
		Dimension:   int32(c.Dim),
		BitWidth:    int32(c.BitWidth),
		RotatorSeed: c.RotatorSeed,
	}
	switch c.Metric {
	case MetricInnerProduct:
		pc.Metric = apiv1.Metric_METRIC_INNER_PRODUCT
	}
	switch c.Variant {
	case VariantMSE:
		pc.Variant = apiv1.QuantizationVariant_QUANTIZATION_VARIANT_MSE
	}
	return pc
}

func collectionStatsToProto(s index.CollectionStats) *apiv1.CollectionStats {
	return &apiv1.CollectionStats{
		VectorCount:         int64(s.VectorCount),
		SealedSegmentCount:  int32(s.SealedSegmentCount),
		GrowingSegmentCount: int32(s.GrowingSegmentCount),
	}
}

func searchResultsToProto(results []index.SearchResult) []*apiv1.SearchResult {
	out := make([]*apiv1.SearchResult, 0, len(results))
	for _, r := range results {
		out = append(out, &apiv1.SearchResult{
			Id:       r.ID,
			Score:    r.Score,
			Metadata: r.Metadata,
		})
	}
	return out
}

// mapError converts engine errors to gRPC status codes.
func mapError(err error) error {
	if err == nil {
		return nil
	}
	switch {
	case errors.Is(err, ErrCollectionNotFound):
		return status.Errorf(codes.NotFound, "%v", err)
	case errors.Is(err, ErrCollectionExists):
		return status.Errorf(codes.AlreadyExists, "%v", err)
	}
	return status.Errorf(codes.Internal, "%v", err)
}
