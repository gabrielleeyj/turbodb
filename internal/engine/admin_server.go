package engine

import (
	"context"
	"fmt"
	"time"

	apiv1 "github.com/gabrielleeyj/turbodb/api/v1"
	"github.com/gabrielleeyj/turbodb/internal/cuda"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// RotatorRegenerateConfirmPhrase must be sent verbatim with a
// RotatorRegenerate request. Regenerating the rotator invalidates every
// quantized code in the collection.
const RotatorRegenerateConfirmPhrase = "i understand this will invalidate all indexes"

// AdminServer implements the TurboDBAdmin gRPC service (SCOPE Task 8.2).
type AdminServer struct {
	apiv1.UnimplementedTurboDBAdminServer

	engine    *Engine
	version   string
	startedAt time.Time
}

// NewAdminServer wraps an engine for administrative RPCs. version is the
// build version string reported by Health.
func NewAdminServer(engine *Engine, version string) *AdminServer {
	return &AdminServer{engine: engine, version: version, startedAt: time.Now()}
}

// Health reports liveness, build version, and process uptime.
func (s *AdminServer) Health(_ context.Context, _ *apiv1.HealthRequest) (*apiv1.HealthResponse, error) {
	return &apiv1.HealthResponse{
		Healthy: true,
		Version: s.version,
		Uptime:  time.Since(s.startedAt).Round(time.Second).String(),
	}, nil
}

// Ready reports whether the engine can serve traffic. The engine finishes
// WAL replay before New returns, so a constructed engine is ready.
func (s *AdminServer) Ready(_ context.Context, _ *apiv1.ReadyRequest) (*apiv1.ReadyResponse, error) {
	if s.engine == nil {
		return &apiv1.ReadyResponse{Ready: false, Reason: "engine not initialized"}, nil
	}
	return &apiv1.ReadyResponse{Ready: true}, nil
}

// GPUInfo reports the CUDA devices visible to the engine. Without CUDA
// support compiled in (or no device present) the device list is empty.
func (s *AdminServer) GPUInfo(_ context.Context, _ *apiv1.GPUInfoRequest) (*apiv1.GPUInfoResponse, error) {
	if !cuda.Available() {
		return &apiv1.GPUInfoResponse{}, nil
	}
	ctx, err := cuda.NewContext(0)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "cuda context: %v", err)
	}
	defer ctx.Close()
	info, err := ctx.DeviceInfo()
	if err != nil {
		return nil, status.Errorf(codes.Internal, "cuda device info: %v", err)
	}
	return &apiv1.GPUInfoResponse{
		Devices: []*apiv1.GPUDevice{{
			Id:               0,
			TotalMemoryBytes: int64(info.TotalBytes), // #nosec G115 -- device memory fits in int64
			FreeMemoryBytes:  int64(info.FreeBytes),  // #nosec G115 -- device memory fits in int64
			ComputeCapability: fmt.Sprintf("%d.%d",
				info.ComputeCapability/10, info.ComputeCapability%10),
		}},
	}, nil
}

// RotatorRegenerate validates the confirmation phrase and rejects the
// operation: sealed segments only retain quantized codes, so re-encoding
// under a new rotator seed requires the raw source vectors. The supported
// path is DROP + re-import (or CDC re-sync from the source of truth).
func (s *AdminServer) RotatorRegenerate(_ context.Context, req *apiv1.RotatorRegenerateRequest) (*apiv1.RotatorRegenerateResponse, error) {
	if req.GetCollection() == "" {
		return nil, status.Error(codes.InvalidArgument, "collection is required")
	}
	if req.GetConfirm() != RotatorRegenerateConfirmPhrase {
		return nil, status.Errorf(codes.InvalidArgument,
			"confirm must be exactly %q", RotatorRegenerateConfirmPhrase)
	}
	if _, err := s.engine.Stats(req.GetCollection()); err != nil {
		return nil, mapError(err)
	}
	return nil, status.Error(codes.Unimplemented,
		"rotator regeneration requires raw source vectors, which sealed segments do not retain; "+
			"drop the collection and re-import (or re-sync via turbodb-sync) instead")
}

// CodebookUpgrade is reserved for future codebook format versions; only v1
// exists today.
func (s *AdminServer) CodebookUpgrade(_ context.Context, req *apiv1.CodebookUpgradeRequest) (*apiv1.CodebookUpgradeResponse, error) {
	if req.GetCollection() == "" {
		return nil, status.Error(codes.InvalidArgument, "collection is required")
	}
	return nil, status.Error(codes.Unimplemented,
		"only codebook version v1 exists; nothing to upgrade")
}
