package replication

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgconn"
)

// PgTableScanner implements SourceScanner against PostgreSQL using keyset
// pagination. The id column is compared and ordered as text with the "C"
// collation, which matches Go's bytewise string ordering — a requirement for
// the reconciler's merge-diff.
type PgTableScanner struct {
	conn *pgconn.PgConn
}

// NewPgTableScanner connects a plain (non-replication) SQL session.
func NewPgTableScanner(ctx context.Context, dsn string) (*PgTableScanner, error) {
	conn, err := pgconn.Connect(ctx, dsn)
	if err != nil {
		return nil, fmt.Errorf("replication: scanner connect: %w", err)
	}
	return &PgTableScanner{conn: conn}, nil
}

// Close terminates the SQL session.
func (s *PgTableScanner) Close() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	return s.conn.Close(ctx)
}

// ScanRows returns up to limit rows with id text strictly greater than
// afterID, ordered bytewise. All values are returned in text format, the
// same representations the CDC path produces, so the shared Transformer
// applies identically.
func (s *PgTableScanner) ScanRows(ctx context.Context, mapping TableMapping, afterID string, limit int) ([]map[string]any, bool, error) {
	table, err := quoteQualifiedIdent(mapping.Postgres)
	if err != nil {
		return nil, false, fmt.Errorf("replication: scanner: %w", err)
	}
	idCol, err := quoteIdent(mapping.Columns.ID)
	if err != nil {
		return nil, false, fmt.Errorf("replication: scanner: %w", err)
	}

	// Fetch limit+1 to learn whether more rows remain.
	sql := fmt.Sprintf(
		`SELECT * FROM %s WHERE (%s::text COLLATE "C") > $1 ORDER BY %s::text COLLATE "C" LIMIT %d`,
		table, idCol, idCol, limit+1)
	rr := s.conn.ExecParams(ctx, sql, [][]byte{[]byte(afterID)}, nil, nil, nil)

	fields := rr.FieldDescriptions()
	names := make([]string, len(fields))
	for i, f := range fields {
		names[i] = f.Name
	}

	var rows []map[string]any
	for rr.NextRow() {
		values := rr.Values()
		row := make(map[string]any, len(values))
		for i, v := range values {
			if v == nil {
				row[names[i]] = nil
			} else {
				row[names[i]] = string(v)
			}
		}
		rows = append(rows, row)
	}
	if _, err := rr.Close(); err != nil {
		return nil, false, fmt.Errorf("replication: scanner: scan %s: %w", mapping.Postgres, err)
	}

	hasMore := len(rows) > limit
	if hasMore {
		rows = rows[:limit]
	}
	return rows, hasMore, nil
}

// quoteQualifiedIdent quotes a schema-qualified name ("public.docs" ->
// "public"."docs"), preventing SQL injection through sync.yaml.
func quoteQualifiedIdent(qualified string) (string, error) {
	parts := strings.Split(qualified, ".")
	if len(parts) != 2 {
		return "", fmt.Errorf("table %q must be schema.table", qualified)
	}
	schema, err := quoteIdent(parts[0])
	if err != nil {
		return "", err
	}
	table, err := quoteIdent(parts[1])
	if err != nil {
		return "", err
	}
	return schema + "." + table, nil
}

// quoteIdent double-quotes a single identifier, doubling embedded quotes.
func quoteIdent(name string) (string, error) {
	if name == "" {
		return "", fmt.Errorf("empty identifier")
	}
	if strings.ContainsRune(name, 0) {
		return "", fmt.Errorf("identifier %q contains NUL", name)
	}
	return `"` + strings.ReplaceAll(name, `"`, `""`) + `"`, nil
}
