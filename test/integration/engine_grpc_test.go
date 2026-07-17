// Package integration exercises the engine end-to-end over its public gRPC
// API, including WAL recovery across an engine restart. CPU-only; no
// external services required.
package integration

import (
	"context"
	"fmt"
	"testing"
	"time"

	apiv1 "github.com/gabrielleeyj/turbodb/api/v1"
	"github.com/gabrielleeyj/turbodb/internal/testutil"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const (
	collection = "docs"
	dim        = 8
	numVectors = 100
)

func dialEngine(t *testing.T, addr string) apiv1.TurboDBEngineClient {
	t.Helper()
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("dial %s: %v", addr, err)
	}
	t.Cleanup(func() { _ = conn.Close() })
	return apiv1.NewTurboDBEngineClient(conn)
}

func testCtx(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)
	return ctx
}

// vecFor returns a deterministic vector dominated by axis i%dim. With 100
// vectors in 8 dimensions many share a dominant axis and 4-bit quantization
// cannot distinguish their small secondary components, so tests assert the
// top result shares the query's dominant axis rather than exact identity.
func vecFor(i int) []float32 {
	v := make([]float32, dim)
	v[i%dim] = 1
	v[(i+1)%dim] = float32(i%97) / 970.0
	return v
}

func idFor(i int) string { return fmt.Sprintf("vec-%03d", i) }

// axisOf extracts i from an id "vec-XXX" and returns its dominant axis.
func axisOf(t *testing.T, id string) int {
	t.Helper()
	var i int
	if _, err := fmt.Sscanf(id, "vec-%d", &i); err != nil {
		t.Fatalf("unexpected id %q: %v", id, err)
	}
	return i % dim
}

func insertAll(ctx context.Context, t *testing.T, client apiv1.TurboDBEngineClient) {
	t.Helper()
	stream, err := client.InsertBatch(ctx)
	if err != nil {
		t.Fatalf("open insert stream: %v", err)
	}
	vectors := make([]*apiv1.Vector, numVectors)
	for i := range vectors {
		vectors[i] = &apiv1.Vector{Id: idFor(i), Values: vecFor(i)}
	}
	if err := stream.Send(&apiv1.InsertBatchRequest{Collection: collection, Vectors: vectors}); err != nil {
		t.Fatalf("send batch: %v", err)
	}
	if _, err := stream.CloseAndRecv(); err != nil {
		t.Fatalf("close insert stream: %v", err)
	}
}

func TestEngineGRPCEndToEnd(t *testing.T) {
	// Arrange: an engine on a persistent data dir so a second instance can
	// recover from its WAL.
	dataDir := t.TempDir()
	addr, stopFirst := testutil.StartEngineWithDataDir(t, dataDir)
	client := dialEngine(t, addr)
	ctx := testCtx(t)

	if _, err := client.CreateCollection(ctx, &apiv1.CreateCollectionRequest{
		Config: &apiv1.CollectionConfig{
			Name:      collection,
			Dimension: dim,
			BitWidth:  4,
			Metric:    apiv1.Metric_METRIC_INNER_PRODUCT,
			Variant:   apiv1.QuantizationVariant_QUANTIZATION_VARIANT_MSE,
		},
	}); err != nil {
		t.Fatalf("create collection: %v", err)
	}

	insertAll(ctx, t, client)

	t.Run("flush seals segments", func(t *testing.T) {
		if _, err := client.Flush(ctx, &apiv1.FlushRequest{Collection: collection}); err != nil {
			t.Fatalf("flush: %v", err)
		}
	})

	t.Run("search finds vectors on the query axis", func(t *testing.T) {
		const probe = 17
		resp, err := client.Search(ctx, &apiv1.SearchRequest{
			Collection: collection,
			Query:      vecFor(probe),
			TopK:       10,
		})
		if err != nil {
			t.Fatalf("search: %v", err)
		}
		results := resp.GetResults()
		if len(results) == 0 {
			t.Fatal("search returned no results")
		}
		if got := axisOf(t, results[0].GetId()); got != probe%dim {
			t.Errorf("best match %s has axis %d, want %d", results[0].GetId(), got, probe%dim)
		}
	})

	t.Run("delete removes vector from results", func(t *testing.T) {
		const victim = 42
		if _, err := client.Delete(ctx, &apiv1.DeleteRequest{
			Collection: collection, Id: idFor(victim),
		}); err != nil {
			t.Fatalf("delete: %v", err)
		}
		resp, err := client.Search(ctx, &apiv1.SearchRequest{
			Collection: collection,
			Query:      vecFor(victim),
			TopK:       10,
		})
		if err != nil {
			t.Fatalf("search after delete: %v", err)
		}
		for _, r := range resp.GetResults() {
			if r.GetId() == idFor(victim) {
				t.Errorf("deleted id %s still in results", idFor(victim))
			}
		}
	})

	t.Run("list ids reflects live count", func(t *testing.T) {
		var count int
		after := ""
		for {
			resp, err := client.ListIDs(ctx, &apiv1.ListIDsRequest{
				Collection: collection, AfterId: after, PageSize: 33,
			})
			if err != nil {
				t.Fatalf("list ids: %v", err)
			}
			count += len(resp.GetIds())
			if !resp.GetHasMore() {
				break
			}
			ids := resp.GetIds()
			after = ids[len(ids)-1]
		}
		if count != numVectors-1 {
			t.Errorf("live ids = %d, want %d", count, numVectors-1)
		}
	})

	t.Run("recovery after restart", func(t *testing.T) {
		// Act: stop the engine, then bring up a fresh instance on the same
		// data dir; WAL replay must restore the collection and vectors.
		stopFirst()
		addr2, _ := testutil.StartEngineWithDataDir(t, dataDir)
		client2 := dialEngine(t, addr2)

		desc, err := client2.DescribeCollection(ctx, &apiv1.DescribeCollectionRequest{Name: collection})
		if err != nil {
			t.Fatalf("describe after restart: %v", err)
		}
		if got := desc.GetConfig().GetDimension(); got != dim {
			t.Errorf("recovered dimension = %d, want %d", got, dim)
		}
		if got := desc.GetStats().GetVectorCount(); got != numVectors-1 {
			t.Errorf("recovered vector count = %d, want %d", got, numVectors-1)
		}

		const probe = 7
		resp, err := client2.Search(ctx, &apiv1.SearchRequest{
			Collection: collection,
			Query:      vecFor(probe),
			TopK:       10,
		})
		if err != nil {
			t.Fatalf("search after restart: %v", err)
		}
		if len(resp.GetResults()) == 0 {
			t.Fatal("post-recovery search returned no results")
		}
		if got := axisOf(t, resp.GetResults()[0].GetId()); got != probe%dim {
			t.Errorf("post-recovery best match %s has axis %d, want %d",
				resp.GetResults()[0].GetId(), got, probe%dim)
		}
	})
}
