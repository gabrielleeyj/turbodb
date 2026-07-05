package engine

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"
)

// AdminHTTPConfig configures the HTTP/JSON admin API (SCOPE Task 8.2).
type AdminHTTPConfig struct {
	// Metrics, when non-nil, is mounted at /metrics.
	Metrics http.Handler
	// RequireMTLSForWrites rejects POST/DELETE requests that did not
	// present a verified client certificate. Enable in production; the
	// server itself must then be run with TLS and client-cert verification.
	RequireMTLSForWrites bool
}

// apiResponse is the envelope for every /api/v1 JSON response.
type apiResponse struct {
	Success bool   `json:"success"`
	Data    any    `json:"data,omitempty"`
	Error   string `json:"error,omitempty"`
}

// collectionJSON mirrors CollectionConfig for the HTTP API.
type collectionJSON struct {
	Name        string `json:"name"`
	Dimension   int    `json:"dimension"`
	BitWidth    int    `json:"bit_width"`
	Metric      string `json:"metric"`
	RotatorSeed uint64 `json:"rotator_seed"`
}

// collectionDetailJSON is the describe payload: config plus stats.
type collectionDetailJSON struct {
	collectionJSON
	VectorCount         int   `json:"vector_count"`
	SealedSegmentCount  int   `json:"sealed_segment_count"`
	GrowingSegmentCount int   `json:"growing_segment_count"`
	HostMemoryBytes     int64 `json:"host_memory_bytes"`
}

// adminHTTP serves the admin API for one engine.
type adminHTTP struct {
	engine *Engine
	cfg    AdminHTTPConfig
}

// NewAdminHTTPHandler builds the admin API handler: /healthz, /readyz,
// /metrics, and /api/v1/collections[...].
func NewAdminHTTPHandler(engine *Engine, cfg AdminHTTPConfig) http.Handler {
	a := &adminHTTP{engine: engine, cfg: cfg}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})
	mux.HandleFunc("GET /readyz", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ready"))
	})
	if cfg.Metrics != nil {
		mux.Handle("GET /metrics", cfg.Metrics)
	}
	mux.HandleFunc("GET /api/v1/collections", a.listCollections)
	mux.HandleFunc("POST /api/v1/collections", a.requireWriteAuth(a.createCollection))
	mux.HandleFunc("GET /api/v1/collections/{name}", a.describeCollection)
	mux.HandleFunc("DELETE /api/v1/collections/{name}", a.requireWriteAuth(a.dropCollection))
	return mux
}

// requireWriteAuth gates mutating endpoints behind a verified client
// certificate when RequireMTLSForWrites is set.
func (a *adminHTTP) requireWriteAuth(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if a.cfg.RequireMTLSForWrites {
			if r.TLS == nil || len(r.TLS.PeerCertificates) == 0 {
				writeJSON(w, http.StatusForbidden, apiResponse{
					Success: false,
					Error:   "write endpoints require a verified client certificate (mTLS)",
				})
				return
			}
		}
		next(w, r)
	}
}

func (a *adminHTTP) listCollections(w http.ResponseWriter, _ *http.Request) {
	configs := a.engine.ListCollections()
	out := make([]collectionJSON, len(configs))
	for i, c := range configs {
		out[i] = toCollectionJSON(c)
	}
	writeJSON(w, http.StatusOK, apiResponse{Success: true, Data: out})
}

func (a *adminHTTP) createCollection(w http.ResponseWriter, r *http.Request) {
	var body collectionJSON
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&body); err != nil {
		writeJSON(w, http.StatusBadRequest, apiResponse{Success: false, Error: "invalid JSON body: " + err.Error()})
		return
	}
	metric := body.Metric
	if metric == "" {
		metric = string(MetricInnerProduct)
	}
	cfg := CollectionConfig{
		Name:        body.Name,
		Dim:         body.Dimension,
		BitWidth:    body.BitWidth,
		Metric:      Metric(metric),
		Variant:     VariantMSE,
		RotatorSeed: body.RotatorSeed,
	}
	if cfg.BitWidth == 0 {
		cfg.BitWidth = 4
	}
	if err := a.engine.CreateCollection(r.Context(), cfg); err != nil {
		writeJSON(w, statusForEngineError(err), apiResponse{Success: false, Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusCreated, apiResponse{Success: true, Data: toCollectionJSON(cfg)})
}

func (a *adminHTTP) describeCollection(w http.ResponseWriter, r *http.Request) {
	cfg, stats, err := a.engine.DescribeCollection(r.PathValue("name"))
	if err != nil {
		writeJSON(w, statusForEngineError(err), apiResponse{Success: false, Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, apiResponse{Success: true, Data: collectionDetailJSON{
		collectionJSON:      toCollectionJSON(cfg),
		VectorCount:         stats.VectorCount,
		SealedSegmentCount:  stats.SealedSegmentCount,
		GrowingSegmentCount: stats.GrowingSegmentCount,
		HostMemoryBytes:     stats.PinnedBytes,
	}})
}

func (a *adminHTTP) dropCollection(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	if err := a.engine.DropCollection(r.Context(), name); err != nil {
		writeJSON(w, statusForEngineError(err), apiResponse{Success: false, Error: err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, apiResponse{Success: true, Data: map[string]string{"dropped": name}})
}

func toCollectionJSON(c CollectionConfig) collectionJSON {
	return collectionJSON{
		Name:        c.Name,
		Dimension:   c.Dim,
		BitWidth:    c.BitWidth,
		Metric:      string(c.Metric),
		RotatorSeed: c.RotatorSeed,
	}
}

// statusForEngineError maps engine errors to HTTP status codes without
// leaking internals: not-found and validation errors are client errors.
func statusForEngineError(err error) int {
	switch {
	case err == nil:
		return http.StatusOK
	case isNotFound(err):
		return http.StatusNotFound
	case isAlreadyExists(err) || isValidation(err):
		return http.StatusBadRequest
	default:
		return http.StatusInternalServerError
	}
}

func isNotFound(err error) bool {
	return errors.Is(err, ErrCollectionNotFound)
}

func isAlreadyExists(err error) bool {
	return strings.Contains(err.Error(), "already exists")
}

func isValidation(err error) bool {
	msg := err.Error()
	return strings.Contains(msg, "required") || strings.Contains(msg, "must be") ||
		strings.Contains(msg, "not supported") || strings.Contains(msg, "invalid")
}

func writeJSON(w http.ResponseWriter, code int, resp apiResponse) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(resp)
}
