package enginerpc

import (
	"context"
	"errors"
	"fmt"
	"net"
	"strings"
	"sync"
	"testing"
	"time"

	apiv1 "github.com/gabrielleeyj/turbodb/api/v1"
	"github.com/gabrielleeyj/turbodb/internal/testutil"
	"github.com/gabrielleeyj/turbodb/pkg/replication"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
)

// fakeEngineServer records requests and returns configured responses.
type fakeEngineServer struct {
	apiv1.UnimplementedTurboDBEngineServer

	mu             sync.Mutex
	insertReqs     []*apiv1.InsertBatchRequest
	deleteReqs     []*apiv1.DeleteRequest
	listReqs       []*apiv1.ListIDsRequest
	insertErr      error
	deleteErrForID string
	listResp       *apiv1.ListIDsResponse
	listErr        error
}

func (f *fakeEngineServer) InsertBatch(stream grpc.ClientStreamingServer[apiv1.InsertBatchRequest, apiv1.InsertBatchResponse]) error {
	for {
		req, err := stream.Recv()
		if err != nil {
			break
		}
		f.mu.Lock()
		f.insertReqs = append(f.insertReqs, req)
		f.mu.Unlock()
	}
	if f.insertErr != nil {
		return f.insertErr
	}
	return stream.SendAndClose(&apiv1.InsertBatchResponse{})
}

func (f *fakeEngineServer) Delete(_ context.Context, req *apiv1.DeleteRequest) (*apiv1.DeleteResponse, error) {
	f.mu.Lock()
	f.deleteReqs = append(f.deleteReqs, req)
	f.mu.Unlock()
	if f.deleteErrForID != "" && req.GetId() == f.deleteErrForID {
		return nil, status.Error(codes.NotFound, "no such id")
	}
	return &apiv1.DeleteResponse{}, nil
}

func (f *fakeEngineServer) ListIDs(_ context.Context, req *apiv1.ListIDsRequest) (*apiv1.ListIDsResponse, error) {
	f.mu.Lock()
	f.listReqs = append(f.listReqs, req)
	f.mu.Unlock()
	if f.listErr != nil {
		return nil, f.listErr
	}
	return f.listResp, nil
}

// dialFake serves fake over bufconn and returns a Client wired to it.
func dialFake(t *testing.T, fake *fakeEngineServer) *Client {
	t.Helper()
	lis := bufconn.Listen(1 << 20)
	srv := grpc.NewServer()
	apiv1.RegisterTurboDBEngineServer(srv, fake)
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)

	conn, err := grpc.NewClient("passthrough:///bufconn",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) {
			return lis.DialContext(ctx)
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("bufconn dial: %v", err)
	}
	client := &Client{conn: conn, client: apiv1.NewTurboDBEngineClient(conn)}
	t.Cleanup(func() { _ = client.Close() })
	return client
}

func testCtx(t *testing.T) context.Context {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	t.Cleanup(cancel)
	return ctx
}

func TestInsertBatchMapsRecordsToVectors(t *testing.T) {
	// Arrange
	fake := &fakeEngineServer{}
	client := dialFake(t, fake)
	records := []replication.VectorRecord{
		{ID: "a", Vector: []float32{1, 2}},
		{ID: "b", Vector: []float32{3, 4}},
	}

	// Act
	err := client.InsertBatch(testCtx(t), "docs", records)

	// Assert
	if err != nil {
		t.Fatalf("InsertBatch: %v", err)
	}
	fake.mu.Lock()
	defer fake.mu.Unlock()
	if len(fake.insertReqs) != 1 {
		t.Fatalf("expected 1 streamed request, got %d", len(fake.insertReqs))
	}
	req := fake.insertReqs[0]
	if req.GetCollection() != "docs" {
		t.Errorf("collection = %q, want docs", req.GetCollection())
	}
	vectors := req.GetVectors()
	if len(vectors) != 2 {
		t.Fatalf("expected 2 vectors, got %d", len(vectors))
	}
	if vectors[0].GetId() != "a" || vectors[1].GetId() != "b" {
		t.Errorf("vector ids = %q, %q", vectors[0].GetId(), vectors[1].GetId())
	}
	if got := vectors[1].GetValues(); len(got) != 2 || got[0] != 3 || got[1] != 4 {
		t.Errorf("vector[1] values = %v, want [3 4]", got)
	}
}

func TestInsertBatchPropagatesServerError(t *testing.T) {
	// Arrange
	fake := &fakeEngineServer{insertErr: status.Error(codes.InvalidArgument, "bad dimension")}
	client := dialFake(t, fake)

	// Act
	err := client.InsertBatch(testCtx(t), "docs", []replication.VectorRecord{{ID: "a", Vector: []float32{1}}})

	// Assert
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "engine insert batch") {
		t.Errorf("error %q lacks context prefix", err)
	}
	if status.Code(errors.Unwrap(err)) != codes.InvalidArgument {
		t.Errorf("unwrapped code = %v, want InvalidArgument", status.Code(errors.Unwrap(err)))
	}
}

func TestDeleteBatchIssuesOneRPCPerID(t *testing.T) {
	// Arrange
	fake := &fakeEngineServer{}
	client := dialFake(t, fake)

	// Act
	err := client.DeleteBatch(testCtx(t), "docs", []string{"a", "b", "c"})

	// Assert
	if err != nil {
		t.Fatalf("DeleteBatch: %v", err)
	}
	fake.mu.Lock()
	defer fake.mu.Unlock()
	if len(fake.deleteReqs) != 3 {
		t.Fatalf("expected 3 delete RPCs, got %d", len(fake.deleteReqs))
	}
	for i, want := range []string{"a", "b", "c"} {
		if fake.deleteReqs[i].GetId() != want || fake.deleteReqs[i].GetCollection() != "docs" {
			t.Errorf("delete[%d] = %s/%s, want docs/%s",
				i, fake.deleteReqs[i].GetCollection(), fake.deleteReqs[i].GetId(), want)
		}
	}
}

func TestDeleteBatchStopsAtFirstError(t *testing.T) {
	// Arrange
	fake := &fakeEngineServer{deleteErrForID: "b"}
	client := dialFake(t, fake)

	// Act
	err := client.DeleteBatch(testCtx(t), "docs", []string{"a", "b", "c"})

	// Assert
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "docs/b") {
		t.Errorf("error %q should name collection/id", err)
	}
	fake.mu.Lock()
	defer fake.mu.Unlock()
	if len(fake.deleteReqs) != 2 {
		t.Errorf("expected early exit after 2 RPCs, got %d", len(fake.deleteReqs))
	}
}

func TestListIDsPassesPagingAndReturnsResult(t *testing.T) {
	tests := []struct {
		name     string
		resp     *apiv1.ListIDsResponse
		listErr  error
		wantIDs  []string
		wantMore bool
		wantErr  bool
	}{
		{
			name:     "page with more",
			resp:     &apiv1.ListIDsResponse{Ids: []string{"x", "y"}, HasMore: true},
			wantIDs:  []string{"x", "y"},
			wantMore: true,
		},
		{
			name: "empty final page",
			resp: &apiv1.ListIDsResponse{},
		},
		{
			name:    "server error",
			listErr: status.Error(codes.NotFound, "no such collection"),
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Arrange
			fake := &fakeEngineServer{listResp: tt.resp, listErr: tt.listErr}
			client := dialFake(t, fake)

			// Act
			ids, hasMore, err := client.ListIDs(testCtx(t), "docs", "after-1", 50)

			// Assert
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error")
				}
				if !strings.Contains(err.Error(), "engine list ids") {
					t.Errorf("error %q lacks context prefix", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("ListIDs: %v", err)
			}
			fake.mu.Lock()
			req := fake.listReqs[0]
			fake.mu.Unlock()
			if req.GetCollection() != "docs" || req.GetAfterId() != "after-1" || req.GetPageSize() != 50 {
				t.Errorf("request = %s/%s/%d, want docs/after-1/50",
					req.GetCollection(), req.GetAfterId(), req.GetPageSize())
			}
			if fmt.Sprint(ids) != fmt.Sprint(tt.wantIDs) {
				t.Errorf("ids = %v, want %v", ids, tt.wantIDs)
			}
			if hasMore != tt.wantMore {
				t.Errorf("hasMore = %v, want %v", hasMore, tt.wantMore)
			}
		})
	}
}

func TestDialRejectsInvalidAddress(t *testing.T) {
	if _, err := Dial("bad\x00addr"); err == nil {
		t.Error("expected dial error for invalid target")
	}
}

// TestClientAgainstRealEngine exercises the adapter end-to-end against an
// in-process engine: create collection, insert, page ids, delete, verify.
func TestClientAgainstRealEngine(t *testing.T) {
	// Arrange
	addr := testutil.StartEngine(t)
	ctx := testCtx(t)

	client, err := Dial(addr)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	t.Cleanup(func() { _ = client.Close() })

	raw, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = raw.Close() })
	if _, err := apiv1.NewTurboDBEngineClient(raw).CreateCollection(ctx, &apiv1.CreateCollectionRequest{
		Config: &apiv1.CollectionConfig{
			Name:      "docs",
			Dimension: 8,
			BitWidth:  4,
			Metric:    apiv1.Metric_METRIC_INNER_PRODUCT,
			Variant:   apiv1.QuantizationVariant_QUANTIZATION_VARIANT_MSE,
		},
	}); err != nil {
		t.Fatalf("create collection: %v", err)
	}

	const n = 10
	records := make([]replication.VectorRecord, n)
	for i := range records {
		vec := make([]float32, 8)
		vec[i%8] = float32(i + 1)
		records[i] = replication.VectorRecord{ID: fmt.Sprintf("id-%02d", i), Vector: vec}
	}

	// Act: insert, then page all ids with a small page size.
	if err := client.InsertBatch(ctx, "docs", records); err != nil {
		t.Fatalf("insert: %v", err)
	}
	var all []string
	after := ""
	for {
		ids, hasMore, err := client.ListIDs(ctx, "docs", after, 3)
		if err != nil {
			t.Fatalf("list ids: %v", err)
		}
		all = append(all, ids...)
		if !hasMore {
			break
		}
		after = ids[len(ids)-1]
	}

	// Assert: all ids present, then delete and verify empty.
	if len(all) != n {
		t.Fatalf("listed %d ids, want %d: %v", len(all), n, all)
	}
	if err := client.DeleteBatch(ctx, "docs", all); err != nil {
		t.Fatalf("delete: %v", err)
	}
	ids, hasMore, err := client.ListIDs(ctx, "docs", "", 100)
	if err != nil {
		t.Fatalf("list after delete: %v", err)
	}
	if len(ids) != 0 || hasMore {
		t.Errorf("expected empty collection, got ids=%v hasMore=%v", ids, hasMore)
	}
}
