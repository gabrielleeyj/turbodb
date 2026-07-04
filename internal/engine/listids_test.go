package engine

import (
	"context"
	"fmt"
	"math/rand/v2"
	"sort"
	"testing"

	apiv1 "github.com/gabrielleeyj/turbodb/api/v1"
	"github.com/gabrielleeyj/turbodb/pkg/index"
)

func TestEngineListIDs(t *testing.T) {
	t.Parallel()
	e := newTestEngine(t)
	ctx := context.Background()
	if err := e.CreateCollection(ctx, defaultCollection("ids")); err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewPCG(1, 2))
	want := make([]string, 0, 20)
	for i := 0; i < 20; i++ {
		id := fmt.Sprintf("v%02d", i)
		want = append(want, id)
		if err := e.Insert(ctx, "ids", index.VectorEntry{ID: id, Values: randVec(rng)}); err != nil {
			t.Fatal(err)
		}
	}
	// Delete two: they must not appear.
	if err := e.Delete(ctx, "ids", "v03"); err != nil {
		t.Fatal(err)
	}
	if err := e.Delete(ctx, "ids", "v17"); err != nil {
		t.Fatal(err)
	}
	want = append(want[:3], want[4:]...)
	want = append(want[:16], want[17:]...)

	ids, err := e.ListIDs("ids")
	if err != nil {
		t.Fatal(err)
	}
	if !sort.StringsAreSorted(ids) {
		t.Error("ids must be sorted")
	}
	if len(ids) != len(want) {
		t.Fatalf("ids: got %d, want %d", len(ids), len(want))
	}
	for i := range want {
		if ids[i] != want[i] {
			t.Errorf("ids[%d]: got %q, want %q", i, ids[i], want[i])
		}
	}

	if _, err := e.ListIDs("nope"); err == nil {
		t.Error("expected error for unknown collection")
	}
}

func TestGRPCListIDsPagination(t *testing.T) {
	t.Parallel()
	e := newTestEngine(t)
	ctx := context.Background()
	if err := e.CreateCollection(ctx, defaultCollection("page")); err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(3, 4))
	for i := 0; i < 25; i++ {
		id := fmt.Sprintf("p%02d", i)
		if err := e.Insert(ctx, "page", index.VectorEntry{ID: id, Values: randVec(rng)}); err != nil {
			t.Fatal(err)
		}
	}
	srv := NewGRPCServer(e)

	var got []string
	after := ""
	pages := 0
	for {
		resp, err := srv.ListIDs(ctx, &apiv1.ListIDsRequest{
			Collection: "page", AfterId: after, PageSize: 10,
		})
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, resp.GetIds()...)
		pages++
		if !resp.GetHasMore() {
			break
		}
		after = resp.GetIds()[len(resp.GetIds())-1]
	}
	if pages != 3 {
		t.Errorf("pages: got %d, want 3", pages)
	}
	if len(got) != 25 {
		t.Fatalf("total ids: got %d, want 25", len(got))
	}
	if !sort.StringsAreSorted(got) {
		t.Error("paged ids must arrive in sorted order")
	}

	// Page size zero uses the default; oversized pages are capped.
	resp, err := srv.ListIDs(ctx, &apiv1.ListIDsRequest{Collection: "page"})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.GetIds()) != 25 || resp.GetHasMore() {
		t.Errorf("default page: got %d ids hasMore=%v", len(resp.GetIds()), resp.GetHasMore())
	}
}
