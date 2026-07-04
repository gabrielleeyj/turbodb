package replication

import (
	"fmt"
	"strconv"
	"strings"
)

// EngineOpKind identifies what the writer should do with a transformed event.
type EngineOpKind int

// Engine operation kinds.
const (
	EngineUpsert EngineOpKind = iota + 1
	EngineDelete
)

// EngineOp is one engine mutation derived from a source change event.
type EngineOp struct {
	Kind       EngineOpKind
	Collection string
	ID         string
	// Vector is set for EngineUpsert only.
	Vector []float32
	// LSN is carried through from the source event for checkpointing.
	LSN uint64
}

// Transformer converts source change events into engine operations using the
// table mappings and filters from sync.yaml.
type Transformer struct {
	cfg *SyncConfig
}

// NewTransformer creates a Transformer from a validated config.
func NewTransformer(cfg *SyncConfig) *Transformer {
	return &Transformer{cfg: cfg}
}

// Transform maps a change event to an engine operation.
//
// The returned bool is false when the event should be skipped: the table is
// not configured, or an insert's row does not match the filter. An update
// whose new row fails the filter becomes a delete — this is what makes
// soft-delete filters like "deleted_at IS NULL" remove rows from the engine
// when they stop matching.
func (t *Transformer) Transform(ev ChangeEvent) (EngineOp, bool, error) {
	m, ok := t.cfg.mappingFor(ev.Table)
	if !ok {
		return EngineOp{}, false, nil
	}

	id, err := extractID(ev.Row, m.Columns.ID)
	if err != nil {
		return EngineOp{}, false, fmt.Errorf("replication: %s %s: %w", ev.Op, ev.Table, err)
	}

	deleteOp := EngineOp{Kind: EngineDelete, Collection: m.Engine, ID: id, LSN: ev.LSN}

	switch ev.Op {
	case OpDelete:
		return deleteOp, true, nil
	case OpInsert, OpUpdate:
		if m.filter != nil && !m.filter.Matches(ev.Row) {
			if ev.Op == OpUpdate {
				return deleteOp, true, nil
			}
			return EngineOp{}, false, nil
		}
		vec, err := extractVector(ev.Row, m.Columns.Embedding)
		if err != nil {
			return EngineOp{}, false, fmt.Errorf("replication: %s %s id=%s: %w", ev.Op, ev.Table, id, err)
		}
		return EngineOp{Kind: EngineUpsert, Collection: m.Engine, ID: id, Vector: vec, LSN: ev.LSN}, true, nil
	default:
		return EngineOp{}, false, fmt.Errorf("replication: %s: unknown op %d", ev.Table, ev.Op)
	}
}

func extractID(row map[string]any, col string) (string, error) {
	v, ok := row[col]
	if !ok || v == nil {
		return "", fmt.Errorf("id column %q missing from row", col)
	}
	switch id := v.(type) {
	case string:
		if id == "" {
			return "", fmt.Errorf("id column %q is empty", col)
		}
		return id, nil
	case int:
		return strconv.Itoa(id), nil
	case int64:
		return strconv.FormatInt(id, 10), nil
	case float64:
		return strconv.FormatFloat(id, 'f', -1, 64), nil
	default:
		return "", fmt.Errorf("id column %q has unsupported type %T", col, v)
	}
}

// extractVector accepts the representations a vector column can arrive in:
// a decoded []float32 / []float64 / []any, or the textual forms produced by
// logical decoding — pgvector's "[0.1,0.2]" and float array "{0.1,0.2}".
func extractVector(row map[string]any, col string) ([]float32, error) {
	v, ok := row[col]
	if !ok || v == nil {
		return nil, fmt.Errorf("embedding column %q missing from row", col)
	}
	switch vec := v.(type) {
	case []float32:
		return validateVector(vec, col)
	case []float64:
		out := make([]float32, len(vec))
		for i, f := range vec {
			out[i] = float32(f)
		}
		return validateVector(out, col)
	case []any:
		out := make([]float32, len(vec))
		for i, e := range vec {
			f, ok := e.(float64)
			if !ok {
				return nil, fmt.Errorf("embedding column %q element %d has type %T", col, i, e)
			}
			out[i] = float32(f)
		}
		return validateVector(out, col)
	case string:
		return parseVectorText(vec, col)
	default:
		return nil, fmt.Errorf("embedding column %q has unsupported type %T", col, v)
	}
}

func parseVectorText(s, col string) ([]float32, error) {
	trimmed := strings.TrimSpace(s)
	if len(trimmed) < 2 {
		return nil, fmt.Errorf("embedding column %q: malformed vector %q", col, s)
	}
	open, closeCh := trimmed[0], trimmed[len(trimmed)-1]
	isVec := open == '[' && closeCh == ']'
	isArr := open == '{' && closeCh == '}'
	if !isVec && !isArr {
		return nil, fmt.Errorf("embedding column %q: malformed vector %q", col, s)
	}
	body := strings.TrimSpace(trimmed[1 : len(trimmed)-1])
	if body == "" {
		return nil, fmt.Errorf("embedding column %q: empty vector", col)
	}
	parts := strings.Split(body, ",")
	out := make([]float32, len(parts))
	for i, p := range parts {
		f, err := strconv.ParseFloat(strings.TrimSpace(p), 32)
		if err != nil {
			return nil, fmt.Errorf("embedding column %q element %d: %w", col, i, err)
		}
		out[i] = float32(f)
	}
	return validateVector(out, col)
}

func validateVector(vec []float32, col string) ([]float32, error) {
	if len(vec) == 0 {
		return nil, fmt.Errorf("embedding column %q: empty vector", col)
	}
	return vec, nil
}
