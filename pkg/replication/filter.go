package replication

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// Filter is a compiled row predicate. Only a small, explicit subset of SQL
// is supported so that filter behavior in the sync pipeline exactly matches
// the operator's intent; anything else is rejected at config load:
//
//	<col> IS NULL
//	<col> IS NOT NULL
//	<col> = <literal>          (string, number, or boolean literal)
//	<col> != <literal>
//	<pred> AND <pred> [...]
type Filter struct {
	preds []predicate
}

type predOp int

const (
	predIsNull predOp = iota + 1
	predIsNotNull
	predEq
	predNeq
)

type predicate struct {
	column string
	op     predOp
	value  any // for predEq / predNeq
}

var (
	reIsNull    = regexp.MustCompile(`(?i)^([a-zA-Z_][a-zA-Z0-9_]*)\s+IS\s+NULL$`)
	reIsNotNull = regexp.MustCompile(`(?i)^([a-zA-Z_][a-zA-Z0-9_]*)\s+IS\s+NOT\s+NULL$`)
	reCompare   = regexp.MustCompile(`^([a-zA-Z_][a-zA-Z0-9_]*)\s*(=|!=|<>)\s*(.+)$`)
	reSplitAnd  = regexp.MustCompile(`(?i)\s+AND\s+`)
)

// ParseFilter compiles a filter expression, failing fast on unsupported syntax.
func ParseFilter(expr string) (*Filter, error) {
	trimmed := strings.TrimSpace(expr)
	if trimmed == "" {
		return nil, fmt.Errorf("empty filter expression")
	}
	parts := reSplitAnd.Split(trimmed, -1)
	preds := make([]predicate, 0, len(parts))
	for _, part := range parts {
		p, err := parsePredicate(strings.TrimSpace(part))
		if err != nil {
			return nil, err
		}
		preds = append(preds, p)
	}
	return &Filter{preds: preds}, nil
}

func parsePredicate(s string) (predicate, error) {
	if m := reIsNotNull.FindStringSubmatch(s); m != nil {
		return predicate{column: m[1], op: predIsNotNull}, nil
	}
	if m := reIsNull.FindStringSubmatch(s); m != nil {
		return predicate{column: m[1], op: predIsNull}, nil
	}
	if m := reCompare.FindStringSubmatch(s); m != nil {
		val, err := parseLiteral(strings.TrimSpace(m[3]))
		if err != nil {
			return predicate{}, fmt.Errorf("predicate %q: %w", s, err)
		}
		op := predEq
		if m[2] != "=" {
			op = predNeq
		}
		return predicate{column: m[1], op: op, value: val}, nil
	}
	return predicate{}, fmt.Errorf("unsupported predicate %q (supported: IS NULL, IS NOT NULL, =, !=, AND)", s)
}

func parseLiteral(s string) (any, error) {
	if len(s) >= 2 && s[0] == '\'' && s[len(s)-1] == '\'' {
		return strings.ReplaceAll(s[1:len(s)-1], "''", "'"), nil
	}
	switch strings.ToLower(s) {
	case "true":
		return true, nil
	case "false":
		return false, nil
	}
	if f, err := strconv.ParseFloat(s, 64); err == nil {
		return f, nil
	}
	return nil, fmt.Errorf("unsupported literal %q (use 'string', number, or true/false)", s)
}

// Matches reports whether the row satisfies every predicate. A column that is
// absent from the row is treated as NULL, which mirrors how a partial replica
// identity presents missing columns.
func (f *Filter) Matches(row map[string]any) bool {
	for _, p := range f.preds {
		if !p.matches(row) {
			return false
		}
	}
	return true
}

func (p predicate) matches(row map[string]any) bool {
	v, ok := row[p.column]
	isNull := !ok || v == nil
	switch p.op {
	case predIsNull:
		return isNull
	case predIsNotNull:
		return !isNull
	case predEq:
		return !isNull && literalEqual(v, p.value)
	case predNeq:
		return !isNull && !literalEqual(v, p.value)
	default:
		return false
	}
}

// literalEqual compares a row value against a parsed literal, tolerating the
// numeric and textual representations produced by logical decoding.
func literalEqual(rowVal, lit any) bool {
	switch want := lit.(type) {
	case string:
		s, ok := rowVal.(string)
		return ok && s == want
	case bool:
		switch got := rowVal.(type) {
		case bool:
			return got == want
		case string:
			// pgoutput renders booleans as "t"/"f".
			return (got == "t" || got == "true") == want
		}
		return false
	case float64:
		switch got := rowVal.(type) {
		case float64:
			return got == want
		case float32:
			return float64(got) == want
		case int:
			return float64(got) == want
		case int64:
			return float64(got) == want
		case string:
			f, err := strconv.ParseFloat(got, 64)
			return err == nil && f == want
		}
		return false
	default:
		return false
	}
}
