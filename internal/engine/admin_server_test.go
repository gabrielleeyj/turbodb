package engine

import (
	"context"
	"strings"
	"testing"

	apiv1 "github.com/gabrielleeyj/turbodb/api/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestAdminHealthAndReady(t *testing.T) {
	t.Parallel()
	e := newTestEngine(t)
	srv := NewAdminServer(e, "v-test")
	ctx := context.Background()

	h, err := srv.Health(ctx, &apiv1.HealthRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if !h.GetHealthy() || h.GetVersion() != "v-test" || h.GetUptime() == "" {
		t.Errorf("health: %+v", h)
	}

	r, err := srv.Ready(ctx, &apiv1.ReadyRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if !r.GetReady() {
		t.Errorf("ready: %+v", r)
	}
}

func TestAdminGPUInfoWithoutCUDA(t *testing.T) {
	t.Parallel()
	// This test binary is built without the cuda tag, so the device list
	// must be empty and the call must not error.
	srv := NewAdminServer(newTestEngine(t), "v-test")
	resp, err := srv.GPUInfo(context.Background(), &apiv1.GPUInfoRequest{})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.GetDevices()) != 0 {
		t.Errorf("devices: %+v", resp.GetDevices())
	}
}

func TestAdminRotatorRegenerate(t *testing.T) {
	t.Parallel()
	e := newTestEngine(t)
	if err := e.CreateCollection(context.Background(), defaultCollection("rot")); err != nil {
		t.Fatal(err)
	}
	srv := NewAdminServer(e, "v-test")
	ctx := context.Background()

	tests := []struct {
		name       string
		req        *apiv1.RotatorRegenerateRequest
		wantCode   codes.Code
		wantSubstr string
	}{
		{"missing collection",
			&apiv1.RotatorRegenerateRequest{Confirm: RotatorRegenerateConfirmPhrase},
			codes.InvalidArgument, "collection is required"},
		{"wrong phrase",
			&apiv1.RotatorRegenerateRequest{Collection: "rot", Confirm: "yes"},
			codes.InvalidArgument, "confirm must be exactly"},
		{"unknown collection",
			&apiv1.RotatorRegenerateRequest{Collection: "nope", Confirm: RotatorRegenerateConfirmPhrase},
			codes.NotFound, ""},
		{"correct phrase is unimplemented",
			&apiv1.RotatorRegenerateRequest{Collection: "rot", Confirm: RotatorRegenerateConfirmPhrase},
			codes.Unimplemented, "raw source vectors"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := srv.RotatorRegenerate(ctx, tt.req)
			st, ok := status.FromError(err)
			if !ok {
				t.Fatalf("not a status error: %v", err)
			}
			if st.Code() != tt.wantCode {
				t.Errorf("code: got %v, want %v (%v)", st.Code(), tt.wantCode, err)
			}
			if tt.wantSubstr != "" && !strings.Contains(st.Message(), tt.wantSubstr) {
				t.Errorf("message %q does not contain %q", st.Message(), tt.wantSubstr)
			}
		})
	}
}

func TestAdminCodebookUpgradeUnimplemented(t *testing.T) {
	t.Parallel()
	srv := NewAdminServer(newTestEngine(t), "v-test")
	_, err := srv.CodebookUpgrade(context.Background(), &apiv1.CodebookUpgradeRequest{Collection: "x"})
	if status.Code(err) != codes.Unimplemented {
		t.Errorf("expected Unimplemented, got %v", err)
	}
	_, err = srv.CodebookUpgrade(context.Background(), &apiv1.CodebookUpgradeRequest{})
	if status.Code(err) != codes.InvalidArgument {
		t.Errorf("expected InvalidArgument for missing collection, got %v", err)
	}
}
