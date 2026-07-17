package replication

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"time"

	"github.com/jackc/pglogrepl"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgproto3"
)

// pgDuplicateObject is the SQLSTATE returned when the replication slot
// already exists; resuming an existing slot is the normal restart path.
const pgDuplicateObject = "42710"

// PgSourceConfig configures a PostgreSQL logical replication source.
type PgSourceConfig struct {
	// DSN is a libpq connection string. "replication=database" is appended
	// automatically if absent.
	DSN string
	// Slot is the replication slot name. Created with the pgoutput plugin
	// if it does not exist.
	Slot string
	// Publication is the PostgreSQL publication to subscribe to.
	Publication string
	// StartLSN is where to resume streaming, normally the checkpointed LSN.
	// Zero starts from the slot's confirmed position.
	StartLSN uint64
	// StandbyTimeout is how often a standby status update is sent even
	// without progress. Default 10s.
	StandbyTimeout time.Duration
}

func (c PgSourceConfig) validate() error {
	if c.DSN == "" {
		return fmt.Errorf("replication: pg source: DSN is required")
	}
	if c.Slot == "" {
		return fmt.Errorf("replication: pg source: slot name is required")
	}
	if !isSafeIdentifier(c.Slot) {
		return fmt.Errorf("replication: pg source: invalid slot name %q (letters, digits, and underscores only)", c.Slot)
	}
	if c.Publication == "" {
		return fmt.Errorf("replication: pg source: publication name is required")
	}
	if !isSafeIdentifier(c.Publication) {
		return fmt.Errorf("replication: pg source: invalid publication name %q (letters, digits, and underscores only)", c.Publication)
	}
	return nil
}

// isSafeIdentifier reports whether s is a plain PostgreSQL identifier
// (letter or underscore, then letters, digits, or underscores). The slot and
// publication names are spliced into replication commands and the pgoutput
// publication_names plugin argument, so anything needing quoting is rejected
// outright rather than escaped.
func isSafeIdentifier(s string) bool {
	for i, r := range s {
		switch {
		case r == '_' || (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z'):
		case r >= '0' && r <= '9':
			if i == 0 {
				return false
			}
		default:
			return false
		}
	}
	return s != ""
}

// PgSource streams change events from a PostgreSQL logical replication slot
// using the pgoutput plugin. It implements EventSource and LSNAcker.
//
// Delivery is at-least-once: PostgreSQL only advances the slot past LSNs
// acknowledged via AckLSN, so a crash between engine flush and checkpoint
// replays the affected transactions. Engine writes must therefore be
// idempotent (upsert semantics / tombstone deletes).
type PgSource struct {
	conn *pgconn.PgConn
	cfg  PgSourceConfig

	relations map[uint32]*pglogrepl.RelationMessage
	// txnEvents buffers the current transaction's events until Commit;
	// queue holds committed events ready for Next.
	txnEvents []ChangeEvent
	queue     []ChangeEvent

	// ackLSN is set by the sync loop (via AckLSN) after a successful flush
	// and checkpoint; it is the position reported to PostgreSQL.
	ackLSN     atomic.Uint64
	lastStatus time.Time
}

// NewPgSource connects, ensures the replication slot exists, and starts
// streaming from cfg.StartLSN.
func NewPgSource(ctx context.Context, cfg PgSourceConfig) (*PgSource, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	if cfg.StandbyTimeout <= 0 {
		cfg.StandbyTimeout = 10 * time.Second
	}

	conn, err := pgconn.Connect(ctx, replicationDSN(cfg.DSN))
	if err != nil {
		return nil, fmt.Errorf("replication: pg connect: %w", err)
	}

	src := &PgSource{
		conn:      conn,
		cfg:       cfg,
		relations: make(map[uint32]*pglogrepl.RelationMessage),
	}
	src.ackLSN.Store(cfg.StartLSN)

	if err := src.start(ctx); err != nil {
		_ = conn.Close(context.Background())
		return nil, err
	}
	return src, nil
}

func replicationDSN(dsn string) string {
	// pgconn accepts both URL and keyword/value DSNs; the keyword form is
	// extended with a space, the URL form with a query parameter.
	if len(dsn) >= 11 && (dsn[:11] == "postgres://" || (len(dsn) >= 13 && dsn[:13] == "postgresql://")) {
		sep := "?"
		for _, ch := range dsn {
			if ch == '?' {
				sep = "&"
				break
			}
		}
		return dsn + sep + "replication=database"
	}
	return dsn + " replication=database"
}

func (s *PgSource) start(ctx context.Context) error {
	_, err := pglogrepl.CreateReplicationSlot(ctx, s.conn, s.cfg.Slot, "pgoutput",
		pglogrepl.CreateReplicationSlotOptions{})
	if err != nil {
		var pgErr *pgconn.PgError
		if !errors.As(err, &pgErr) || pgErr.Code != pgDuplicateObject {
			return fmt.Errorf("replication: create slot %q: %w", s.cfg.Slot, err)
		}
	}

	opts := pglogrepl.StartReplicationOptions{PluginArgs: []string{
		"proto_version '1'",
		fmt.Sprintf("publication_names '%s'", s.cfg.Publication),
	}}
	start := pglogrepl.LSN(s.cfg.StartLSN)
	if err := pglogrepl.StartReplication(ctx, s.conn, s.cfg.Slot, start, opts); err != nil {
		return fmt.Errorf("replication: start replication on slot %q: %w", s.cfg.Slot, err)
	}
	s.lastStatus = time.Now()
	return nil
}

// AckLSN records the highest LSN that has been flushed to the engine and
// checkpointed. It is reported to PostgreSQL on the next status update,
// allowing WAL up to that point to be recycled.
func (s *PgSource) AckLSN(lsn uint64) { s.ackLSN.Store(lsn) }

// Next returns the next change event, blocking until one arrives or ctx is
// done. Keepalives and standby status updates are handled internally — also
// while draining a large buffered transaction, so a slow consumer never
// starves the walsender past wal_sender_timeout.
func (s *PgSource) Next(ctx context.Context) (ChangeEvent, error) {
	for {
		if len(s.queue) > 0 {
			if time.Since(s.lastStatus) >= s.cfg.StandbyTimeout {
				if err := s.sendStatus(ctx); err != nil {
					return ChangeEvent{}, err
				}
			}
			ev := s.queue[0]
			s.queue = s.queue[1:]
			return ev, nil
		}
		if err := s.pump(ctx); err != nil {
			return ChangeEvent{}, err
		}
	}
}

// pump receives one protocol message, sending standby status updates as
// needed, and appends any decoded change events to the queue.
func (s *PgSource) pump(ctx context.Context) error {
	if time.Since(s.lastStatus) >= s.cfg.StandbyTimeout {
		if err := s.sendStatus(ctx); err != nil {
			return err
		}
	}

	rcvCtx, cancel := context.WithDeadline(ctx, s.lastStatus.Add(s.cfg.StandbyTimeout))
	rawMsg, err := s.conn.ReceiveMessage(rcvCtx)
	cancel()
	if err != nil {
		if pgconn.Timeout(err) && ctx.Err() == nil {
			// Idle interval elapsed: send a status update and keep waiting.
			return s.sendStatus(ctx)
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return fmt.Errorf("replication: receive: %w", err)
	}

	switch msg := rawMsg.(type) {
	case *pgproto3.ErrorResponse:
		return fmt.Errorf("replication: postgres error: %s (%s)", msg.Message, msg.Code)
	case *pgproto3.CopyData:
		return s.handleCopyData(ctx, msg.Data)
	default:
		return nil // NoticeResponse etc. — ignore
	}
}

func (s *PgSource) handleCopyData(ctx context.Context, data []byte) error {
	if len(data) == 0 {
		return nil
	}
	switch data[0] {
	case pglogrepl.PrimaryKeepaliveMessageByteID:
		ka, err := pglogrepl.ParsePrimaryKeepaliveMessage(data[1:])
		if err != nil {
			return fmt.Errorf("replication: parse keepalive: %w", err)
		}
		if ka.ReplyRequested {
			return s.sendStatus(ctx)
		}
		return nil
	case pglogrepl.XLogDataByteID:
		xld, err := pglogrepl.ParseXLogData(data[1:])
		if err != nil {
			return fmt.Errorf("replication: parse xlog data: %w", err)
		}
		return s.handleWALMessage(xld.WALData)
	default:
		return nil
	}
}

func (s *PgSource) sendStatus(ctx context.Context) error {
	pos := pglogrepl.LSN(s.ackLSN.Load())
	err := pglogrepl.SendStandbyStatusUpdate(ctx, s.conn, pglogrepl.StandbyStatusUpdate{
		WALWritePosition: pos,
		WALFlushPosition: pos,
		WALApplyPosition: pos,
	})
	if err != nil {
		return fmt.Errorf("replication: standby status update: %w", err)
	}
	s.lastStatus = time.Now()
	return nil
}

// Close terminates the replication connection.
func (s *PgSource) Close() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	return s.conn.Close(ctx)
}
