package engine

import (
	"context"
	"errors"
	"io"
	"math/rand/v2"
	"net"
	"path/filepath"
	"testing"
	"time"

	apiv1 "github.com/gabrielleeyj/turbodb/api/v1"
	"github.com/gabrielleeyj/turbodb/pkg/index"
	"github.com/gabrielleeyj/turbodb/pkg/search"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
)

// testDim and testBitWidth match a precomputed codebook (pkg/codebook/precomputed/).
const (
	testDim      = 128
	testBitWidth = 4
)

func newTestEngine(t *testing.T) *Engine {
	t.Helper()
	dir := t.TempDir()
	e, err := New(EngineConfig{DataDir: dir})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { _ = e.Close() })
	return e
}

func defaultCollection(name string) CollectionConfig {
	return CollectionConfig{
		Name:        name,
		Dim:         testDim,
		BitWidth:    testBitWidth,
		Metric:      MetricInnerProduct,
		Variant:     VariantMSE,
		RotatorSeed: 42,
	}
}

func randVec(rng *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rng.Float32()*2 - 1
	}
	return v
}

func TestCollectionLifecycle(t *testing.T) {
	t.Parallel()
	e := newTestEngine(t)

	cfg := defaultCollection("vectors")
	if err := e.CreateCollection(context.Background(), cfg); err != nil {
		t.Fatalf("Create: %v", err)
	}

	// Duplicate should error.
	err := e.CreateCollection(context.Background(), cfg)
	if !errors.Is(err, ErrCollectionExists) {
		t.Fatalf("expected ErrCollectionExists, got %v", err)
	}

	list := e.ListCollections()
	if len(list) != 1 || list[0].Name != "vectors" {
		t.Fatalf("ListCollections: got %+v", list)
	}

	got, _, err := e.DescribeCollection("vectors")
	if err != nil {
		t.Fatal(err)
	}
	if got.Dim != testDim {
		t.Errorf("Dim = %d, want %d", got.Dim, testDim)
	}

	if err := e.DropCollection(context.Background(), "vectors"); err != nil {
		t.Fatal(err)
	}
	if got := e.ListCollections(); len(got) != 0 {
		t.Errorf("ListCollections after drop: %+v", got)
	}
}

func TestInsertSearchDelete(t *testing.T) {
	t.Parallel()
	e := newTestEngine(t)
	ctx := context.Background()

	if err := e.CreateCollection(ctx, defaultCollection("c")); err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewPCG(7, 11))
	for i := range 100 {
		err := e.Insert(ctx, "c", index.VectorEntry{
			ID:     idOf(i),
			Values: randVec(rng, testDim),
		})
		if err != nil {
			t.Fatalf("Insert %d: %v", i, err)
		}
	}

	// Search using the first inserted vector — it should appear at top.
	rng2 := rand.New(rand.NewPCG(7, 11))
	query := randVec(rng2, testDim)
	results, plan, err := e.Search(ctx, "c", query, search.Options{TopK: 5})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 5 {
		t.Errorf("results len = %d, want 5", len(results))
	}
	if results[0].ID != idOf(0) {
		t.Errorf("top result id = %q, want %q", results[0].ID, idOf(0))
	}
	if plan.EffectiveTopK != 5 {
		t.Errorf("plan.EffectiveTopK = %d, want 5", plan.EffectiveTopK)
	}

	// Delete the top result and verify it's gone.
	if err := e.Delete(ctx, "c", idOf(0)); err != nil {
		t.Fatal(err)
	}
	results, _, err = e.Search(ctx, "c", query, search.Options{TopK: 5})
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range results {
		if r.ID == idOf(0) {
			t.Errorf("deleted id %q still in results", r.ID)
		}
	}
}

func TestRecoveryReplaysWAL(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	ctx := context.Background()

	e1, err := New(EngineConfig{DataDir: dir})
	if err != nil {
		t.Fatal(err)
	}
	if err := e1.CreateCollection(ctx, defaultCollection("c")); err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(1, 2))
	for i := range 25 {
		if err := e1.Insert(ctx, "c", index.VectorEntry{ID: idOf(i), Values: randVec(rng, testDim)}); err != nil {
			t.Fatal(err)
		}
	}
	if err := e1.Delete(ctx, "c", idOf(3)); err != nil {
		t.Fatal(err)
	}
	if err := e1.Close(); err != nil {
		t.Fatal(err)
	}

	// Reopen — recovery should replay everything.
	e2, err := New(EngineConfig{DataDir: dir})
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer e2.Close()

	stats, err := e2.Stats("c")
	if err != nil {
		t.Fatal(err)
	}
	if stats.VectorCount != 25 {
		t.Errorf("VectorCount = %d, want 25", stats.VectorCount)
	}
	if stats.TombstoneCount != 1 {
		t.Errorf("TombstoneCount = %d, want 1", stats.TombstoneCount)
	}
}

func TestSearchPlannerOptions(t *testing.T) {
	t.Parallel()
	e := newTestEngine(t)
	ctx := context.Background()

	if err := e.CreateCollection(ctx, defaultCollection("p")); err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(3, 5))
	for i := range 60 {
		if err := e.Insert(ctx, "p", index.VectorEntry{ID: idOf(i), Values: randVec(rng, testDim)}); err != nil {
			t.Fatal(err)
		}
	}

	// Invalid options bubble up from the planner.
	if _, _, err := e.Search(ctx, "p", make([]float32, testDim), search.Options{TopK: 0}); err == nil {
		t.Errorf("expected error for top_k=0")
	}

	// Oversearch widens the per-segment candidate pool.
	rng2 := rand.New(rand.NewPCG(3, 5))
	q := randVec(rng2, testDim)
	_, plan, err := e.Search(ctx, "p", q, search.Options{TopK: 5, OversearchFactor: 3.0})
	if err != nil {
		t.Fatal(err)
	}
	if plan.EffectiveTopK != 15 {
		t.Errorf("plan.EffectiveTopK = %d, want 15", plan.EffectiveTopK)
	}
	if plan.SegmentsSearched < 1 {
		t.Errorf("plan.SegmentsSearched = %d, want >= 1", plan.SegmentsSearched)
	}

	// Rerank without a configured reranker is a no-op.
	_, plan, err = e.Search(ctx, "p", q, search.Options{TopK: 3, Rerank: true})
	if err != nil {
		t.Fatal(err)
	}
	if plan.Reranked {
		t.Errorf("plan.Reranked = true with no reranker configured")
	}
}

func TestErrorMapping(t *testing.T) {
	t.Parallel()
	e := newTestEngine(t)
	ctx := context.Background()

	// Search nonexistent collection.
	_, _, err := e.Search(ctx, "missing", []float32{1, 2, 3}, search.Options{TopK: 1})
	if !errors.Is(err, ErrCollectionNotFound) {
		t.Errorf("expected ErrCollectionNotFound, got %v", err)
	}
}

func TestValidation(t *testing.T) {
	t.Parallel()
	e := newTestEngine(t)
	ctx := context.Background()

	cases := []struct {
		name string
		cfg  CollectionConfig
	}{
		{"empty name", CollectionConfig{Dim: testDim, BitWidth: testBitWidth, Metric: MetricInnerProduct, Variant: VariantMSE}},
		{"bad dim", CollectionConfig{Name: "x", Dim: 0, BitWidth: 4, Metric: MetricInnerProduct, Variant: VariantMSE}},
		{"bad bitwidth", CollectionConfig{Name: "x", Dim: 128, BitWidth: 9, Metric: MetricInnerProduct, Variant: VariantMSE}},
		{"missing variant", CollectionConfig{Name: "x", Dim: 128, BitWidth: 4, Metric: MetricInnerProduct}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := e.CreateCollection(ctx, tc.cfg); err == nil {
				t.Errorf("expected validation error for %s", tc.name)
			}
		})
	}
}

func TestGRPCRoundTrip(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	eng, err := New(EngineConfig{DataDir: filepath.Join(dir, "data")})
	if err != nil {
		t.Fatal(err)
	}
	defer eng.Close()

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := grpc.NewServer()
	apiv1.RegisterTurboDBEngineServer(srv, NewGRPCServer(eng))

	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.GracefulStop)

	conn, err := grpc.NewClient(lis.Addr().String(), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()

	client := apiv1.NewTurboDBEngineClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Create collection.
	_, err = client.CreateCollection(ctx, &apiv1.CreateCollectionRequest{
		Config: &apiv1.CollectionConfig{
			Name:        "grpc-test",
			Dimension:   testDim,
			BitWidth:    testBitWidth,
			Metric:      apiv1.Metric_METRIC_INNER_PRODUCT,
			Variant:     apiv1.QuantizationVariant_QUANTIZATION_VARIANT_MSE,
			RotatorSeed: 7,
		},
	})
	if err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}

	// Duplicate → AlreadyExists.
	_, err = client.CreateCollection(ctx, &apiv1.CreateCollectionRequest{
		Config: &apiv1.CollectionConfig{
			Name:      "grpc-test",
			Dimension: testDim,
			BitWidth:  testBitWidth,
			Metric:    apiv1.Metric_METRIC_INNER_PRODUCT,
			Variant:   apiv1.QuantizationVariant_QUANTIZATION_VARIANT_MSE,
		},
	})
	if status.Code(err) != codes.AlreadyExists {
		t.Errorf("expected AlreadyExists, got %v", err)
	}

	// Insert via streaming InsertBatch.
	batch, err := client.InsertBatch(ctx)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(99, 100))
	const n = 30
	vecs := make([]*apiv1.Vector, n)
	for i := range n {
		vecs[i] = &apiv1.Vector{Id: idOf(i), Values: randVec(rng, testDim)}
	}
	if err := batch.Send(&apiv1.InsertBatchRequest{Collection: "grpc-test", Vectors: vecs}); err != nil {
		t.Fatal(err)
	}
	resp, err := batch.CloseAndRecv()
	if err != nil {
		t.Fatal(err)
	}
	if resp.GetInsertedCount() != n {
		t.Errorf("InsertedCount = %d, want %d", resp.GetInsertedCount(), n)
	}

	// Search — top result should match the first vector when we re-derive it.
	rng2 := rand.New(rand.NewPCG(99, 100))
	query := randVec(rng2, testDim)
	searchResp, err := client.Search(ctx, &apiv1.SearchRequest{
		Collection: "grpc-test",
		Query:      query,
		TopK:       3,
	})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(searchResp.GetResults()) != 3 {
		t.Errorf("results len = %d, want 3", len(searchResp.GetResults()))
	}
	if searchResp.GetResults()[0].GetId() != idOf(0) {
		t.Errorf("top id = %q, want %q", searchResp.GetResults()[0].GetId(), idOf(0))
	}

	// Search nonexistent → NotFound.
	_, err = client.Search(ctx, &apiv1.SearchRequest{Collection: "missing", Query: query, TopK: 1})
	if status.Code(err) != codes.NotFound {
		t.Errorf("expected NotFound, got %v", err)
	}

	// SearchBatch (server-streaming).
	stream, err := client.SearchBatch(ctx, &apiv1.SearchBatchRequest{
		Queries: []*apiv1.SearchRequest{
			{Collection: "grpc-test", Query: query, TopK: 1},
			{Collection: "grpc-test", Query: query, TopK: 2},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	var streamCount int
	for {
		_, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatalf("SearchBatch recv: %v", err)
		}
		streamCount++
	}
	if streamCount != 2 {
		t.Errorf("SearchBatch stream count = %d, want 2", streamCount)
	}

	// Stats.
	statsResp, err := client.GetStats(ctx, &apiv1.GetStatsRequest{Collection: "grpc-test"})
	if err != nil {
		t.Fatal(err)
	}
	if statsResp.GetStats().GetVectorCount() != n {
		t.Errorf("Stats.VectorCount = %d, want %d", statsResp.GetStats().GetVectorCount(), n)
	}
}

func idOf(i int) string {
	return fmtIntPad(i)
}

// fmtIntPad zero-pads i to width 6 for stable lexicographic ordering.
func fmtIntPad(i int) string {
	const digits = "0123456789"
	if i == 0 {
		return "id-000000"
	}
	buf := []byte("id-000000")
	pos := len(buf) - 1
	for i > 0 && pos >= 3 {
		buf[pos] = digits[i%10]
		i /= 10
		pos--
	}
	return string(buf)
}
