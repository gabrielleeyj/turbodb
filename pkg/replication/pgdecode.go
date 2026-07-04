package replication

import (
	"fmt"

	"github.com/jackc/pglogrepl"
)

// handleWALMessage decodes one pgoutput message. Row events are buffered per
// transaction and released to the queue only when the Commit message arrives,
// stamped with the transaction's end LSN. Checkpointing that LSN and resuming
// from it therefore never replays the transaction (resuming from the commit
// record's own LSN would), and consumers never observe partial transactions.
func (s *PgSource) handleWALMessage(walData []byte) error {
	msg, err := pglogrepl.Parse(walData)
	if err != nil {
		return fmt.Errorf("replication: parse pgoutput message: %w", err)
	}

	switch m := msg.(type) {
	case *pglogrepl.BeginMessage:
		s.txnEvents = s.txnEvents[:0]
	case *pglogrepl.CommitMessage:
		endLSN := uint64(m.TransactionEndLSN)
		for i := range s.txnEvents {
			s.txnEvents[i].LSN = endLSN
		}
		s.queue = append(s.queue, s.txnEvents...)
		s.txnEvents = s.txnEvents[:0]
	case *pglogrepl.RelationMessage:
		// Relation messages (re)announce a table's schema; pgoutput sends
		// one before first use and again after DDL, which is what keeps
		// column-name resolution correct across schema evolution.
		s.relations[m.RelationID] = m
	case *pglogrepl.InsertMessage:
		return s.bufferTuple(OpInsert, m.RelationID, m.Tuple)
	case *pglogrepl.UpdateMessage:
		return s.bufferTuple(OpUpdate, m.RelationID, m.NewTuple)
	case *pglogrepl.DeleteMessage:
		return s.bufferTuple(OpDelete, m.RelationID, m.OldTuple)
	}
	// TruncateMessage, TypeMessage, OriginMessage: not replicated.
	return nil
}

func (s *PgSource) bufferTuple(op Op, relationID uint32, tuple *pglogrepl.TupleData) error {
	rel, ok := s.relations[relationID]
	if !ok {
		return fmt.Errorf("replication: %s for unknown relation OID %d (no Relation message seen)", op, relationID)
	}
	if tuple == nil {
		return fmt.Errorf("replication: %s on %s.%s has no tuple data (check REPLICA IDENTITY)",
			op, rel.Namespace, rel.RelationName)
	}
	row, err := decodeTuple(rel, tuple)
	if err != nil {
		return fmt.Errorf("replication: %s on %s.%s: %w", op, rel.Namespace, rel.RelationName, err)
	}
	s.txnEvents = append(s.txnEvents, ChangeEvent{
		Op:    op,
		Table: rel.Namespace + "." + rel.RelationName,
		Row:   row,
		// LSN is stamped with the transaction end LSN at commit.
	})
	return nil
}

// decodeTuple maps tuple columns to a name -> value row. Text-format values
// come through as strings (the transformer parses vectors, numbers, and
// booleans from their text renderings); NULLs are nil; unchanged TOAST
// values ('u') are omitted so they read as absent rather than wrong.
func decodeTuple(rel *pglogrepl.RelationMessage, tuple *pglogrepl.TupleData) (map[string]any, error) {
	if len(tuple.Columns) != len(rel.Columns) {
		return nil, fmt.Errorf("tuple has %d columns, relation has %d", len(tuple.Columns), len(rel.Columns))
	}
	row := make(map[string]any, len(tuple.Columns))
	for i, col := range tuple.Columns {
		name := rel.Columns[i].Name
		switch col.DataType {
		case pglogrepl.TupleDataTypeText:
			row[name] = string(col.Data)
		case pglogrepl.TupleDataTypeNull:
			row[name] = nil
		case pglogrepl.TupleDataTypeToast:
			// Unchanged TOASTed value: not sent. Leave absent.
		case pglogrepl.TupleDataTypeBinary:
			return nil, fmt.Errorf("column %q arrived in binary format; pgoutput binary mode is not supported", name)
		default:
			return nil, fmt.Errorf("column %q has unknown tuple data type %q", name, col.DataType)
		}
	}
	return row, nil
}
