package engine

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func newAdminAPI(t *testing.T, cfg AdminHTTPConfig) http.Handler {
	t.Helper()
	return NewAdminHTTPHandler(newTestEngine(t), cfg)
}

func doJSON(t *testing.T, h http.Handler, method, path, body string) (int, apiResponse) {
	t.Helper()
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	var resp apiResponse
	if rec.Body.Len() > 0 && strings.Contains(rec.Header().Get("Content-Type"), "json") {
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
			t.Fatalf("decode %s %s response: %v (%s)", method, path, err, rec.Body.String())
		}
	}
	return rec.Code, resp
}

func TestAdminHTTPProbes(t *testing.T) {
	t.Parallel()
	h := newAdminAPI(t, AdminHTTPConfig{})

	for _, path := range []string{"/healthz", "/readyz"} {
		req := httptest.NewRequest(http.MethodGet, path, nil)
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Errorf("%s: status %d", path, rec.Code)
		}
	}
}

func TestAdminHTTPCollectionCRUD(t *testing.T) {
	t.Parallel()
	h := newAdminAPI(t, AdminHTTPConfig{})

	// Empty list.
	code, resp := doJSON(t, h, http.MethodGet, "/api/v1/collections", "")
	if code != http.StatusOK || !resp.Success {
		t.Fatalf("list: %d %+v", code, resp)
	}

	// Create.
	code, resp = doJSON(t, h, http.MethodPost, "/api/v1/collections",
		`{"name":"docs","dimension":8,"bit_width":4}`)
	if code != http.StatusCreated || !resp.Success {
		t.Fatalf("create: %d %+v", code, resp)
	}

	// Duplicate create is a client error.
	code, resp = doJSON(t, h, http.MethodPost, "/api/v1/collections",
		`{"name":"docs","dimension":8,"bit_width":4}`)
	if code != http.StatusBadRequest || resp.Success {
		t.Errorf("duplicate create: %d %+v", code, resp)
	}

	// Unknown JSON fields are rejected.
	code, _ = doJSON(t, h, http.MethodPost, "/api/v1/collections",
		`{"name":"x","dimension":8,"sneaky":true}`)
	if code != http.StatusBadRequest {
		t.Errorf("unknown field: %d", code)
	}

	// Describe.
	code, resp = doJSON(t, h, http.MethodGet, "/api/v1/collections/docs", "")
	if code != http.StatusOK || !resp.Success {
		t.Fatalf("describe: %d %+v", code, resp)
	}
	detail, _ := resp.Data.(map[string]any)
	if detail["dimension"].(float64) != 8 || detail["vector_count"].(float64) != 0 {
		t.Errorf("describe payload: %+v", detail)
	}

	// Describe unknown -> 404 with envelope.
	code, resp = doJSON(t, h, http.MethodGet, "/api/v1/collections/nope", "")
	if code != http.StatusNotFound || resp.Success || resp.Error == "" {
		t.Errorf("describe unknown: %d %+v", code, resp)
	}

	// Drop, then the list is empty again.
	code, resp = doJSON(t, h, http.MethodDelete, "/api/v1/collections/docs", "")
	if code != http.StatusOK || !resp.Success {
		t.Fatalf("drop: %d %+v", code, resp)
	}
	code, resp = doJSON(t, h, http.MethodGet, "/api/v1/collections", "")
	if code != http.StatusOK {
		t.Fatal(code)
	}
	if data, ok := resp.Data.([]any); ok && len(data) != 0 {
		t.Errorf("list after drop: %+v", resp.Data)
	}
}

func TestAdminHTTPWriteGateRequiresMTLS(t *testing.T) {
	t.Parallel()
	h := newAdminAPI(t, AdminHTTPConfig{RequireMTLSForWrites: true})

	// Reads pass without a client certificate.
	code, _ := doJSON(t, h, http.MethodGet, "/api/v1/collections", "")
	if code != http.StatusOK {
		t.Errorf("read without cert: %d", code)
	}

	// Writes without a client certificate are rejected.
	code, resp := doJSON(t, h, http.MethodPost, "/api/v1/collections",
		`{"name":"docs","dimension":8}`)
	if code != http.StatusForbidden || resp.Success {
		t.Errorf("write without cert: %d %+v", code, resp)
	}

	// Writes with a verified peer certificate pass the gate.
	req := httptest.NewRequest(http.MethodPost, "/api/v1/collections",
		strings.NewReader(`{"name":"docs","dimension":8}`))
	req.TLS = &tls.ConnectionState{PeerCertificates: []*x509.Certificate{{}}}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusCreated {
		t.Errorf("write with cert: %d (%s)", rec.Code, rec.Body.String())
	}
}
