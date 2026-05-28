package pgproto

import (
	"errors"
	"fmt"
	"net"
)

// Client is a Go reference implementation of the IPC client. The production
// client lives in the C extension; this one drives tests and tooling.
type Client struct {
	conn net.Conn
}

// Dial connects to the engine IPC socket.
func Dial(socketPath string) (*Client, error) {
	conn, err := net.Dial("unix", socketPath)
	if err != nil {
		return nil, fmt.Errorf("pgproto: dial %s: %w", socketPath, err)
	}
	return &Client{conn: conn}, nil
}

// roundtrip sends a frame and returns the reply frame, mapping OpError to a
// Go error.
func (c *Client) roundtrip(op Opcode, payload []byte) (Frame, error) {
	if err := WriteFrame(c.conn, op, payload); err != nil {
		return Frame{}, err
	}
	reply, err := ReadFrame(c.conn)
	if err != nil {
		return Frame{}, err
	}
	if reply.Opcode == OpError {
		return Frame{}, errors.New(DecodeError(reply.Payload))
	}
	return reply, nil
}

// expectAck sends a frame and requires an ACK reply.
func (c *Client) expectAck(op Opcode, payload []byte) error {
	reply, err := c.roundtrip(op, payload)
	if err != nil {
		return err
	}
	if reply.Opcode != OpAck {
		return fmt.Errorf("pgproto: expected ACK, got %s", reply.Opcode)
	}
	return nil
}

// BuildBegin starts an index build.
func (c *Client) BuildBegin(m BuildBegin) error { return c.expectAck(OpBuildBegin, m.Encode()) }

// BuildVector sends one vector during a build.
func (c *Client) BuildVector(m VectorMsg) error { return c.expectAck(OpBuildVector, m.Encode()) }

// Insert adds a vector.
func (c *Client) Insert(m VectorMsg) error { return c.expectAck(OpInsert, m.Encode()) }

// Delete tombstones a tid.
func (c *Client) Delete(m DeleteMsg) error { return c.expectAck(OpDelete, m.Encode()) }

// BuildCommit finalizes a build.
func (c *Client) BuildCommit() error { return c.expectAck(OpBuildCommit, nil) }

// Search runs a query and drains the result stream.
func (c *Client) Search(m SearchBegin) ([]ResultMsg, error) {
	if err := c.expectAck(OpSearchBegin, m.Encode()); err != nil {
		return nil, err
	}
	var results []ResultMsg
	for {
		reply, err := c.roundtrip(OpSearchNext, nil)
		if err != nil {
			return nil, err
		}
		if reply.Opcode != OpResult {
			return nil, fmt.Errorf("pgproto: expected RESULT, got %s", reply.Opcode)
		}
		row, derr := DecodeResultMsg(reply.Payload)
		if derr != nil {
			return nil, derr
		}
		if row.Done {
			break
		}
		results = append(results, row)
	}
	if err := c.expectAck(OpSearchEnd, nil); err != nil {
		return nil, err
	}
	return results, nil
}

// Stats fetches collection statistics.
func (c *Client) Stats() (StatsReply, error) {
	reply, err := c.roundtrip(OpStats, nil)
	if err != nil {
		return StatsReply{}, err
	}
	if reply.Opcode != OpStats {
		return StatsReply{}, fmt.Errorf("pgproto: expected STATS, got %s", reply.Opcode)
	}
	return DecodeStatsReply(reply.Payload)
}

// Shutdown asks the engine to close this connection.
func (c *Client) Shutdown() error { return c.expectAck(OpShutdown, nil) }

// Close closes the connection.
func (c *Client) Close() error { return c.conn.Close() }
