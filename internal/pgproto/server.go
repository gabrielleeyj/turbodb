package pgproto

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"os"
	"path/filepath"
	"sync"
)

// DefaultSocketPath is the default Unix socket address for the engine IPC.
const DefaultSocketPath = "/var/run/turbodb/engine.sock"

// ServerConfig configures the IPC server.
type ServerConfig struct {
	// SocketPath is the Unix socket to listen on.
	SocketPath string
	// AllowedUID, when >= 0, restricts connections to peers with this UID
	// (verified via SO_PEERCRED / LOCAL_PEERCRED). -1 disables the check.
	AllowedUID int
	// Logger receives operational logs; defaults to slog.Default().
	Logger *slog.Logger
}

// Server accepts IPC connections and dispatches frames to a Handler.
type Server struct {
	cfg     ServerConfig
	handler Handler
	logger  *slog.Logger

	mu       sync.Mutex
	listener net.Listener
	conns    map[net.Conn]struct{}
	closed   bool
}

// NewServer constructs an IPC server. AllowedUID defaults to -1 (no check) when
// the zero value is undesirable; callers should set it explicitly.
func NewServer(handler Handler, cfg ServerConfig) *Server {
	logger := cfg.Logger
	if logger == nil {
		logger = slog.Default()
	}
	if cfg.SocketPath == "" {
		cfg.SocketPath = DefaultSocketPath
	}
	return &Server{
		cfg:     cfg,
		handler: handler,
		logger:  logger,
		conns:   make(map[net.Conn]struct{}),
	}
}

// Listen binds the Unix socket, creating its parent directory and removing any
// stale socket file.
func (s *Server) Listen() error {
	if dir := filepath.Dir(s.cfg.SocketPath); dir != "." && dir != "/" {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return fmt.Errorf("pgproto: create socket dir: %w", err)
		}
	}
	if err := os.Remove(s.cfg.SocketPath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("pgproto: remove stale socket: %w", err)
	}
	ln, err := net.Listen("unix", s.cfg.SocketPath)
	if err != nil {
		return fmt.Errorf("pgproto: listen on %s: %w", s.cfg.SocketPath, err)
	}
	s.mu.Lock()
	s.listener = ln
	s.mu.Unlock()
	return nil
}

// Serve accepts connections until the context is cancelled or the server is
// closed. It blocks; run it in a goroutine.
func (s *Server) Serve(ctx context.Context) error {
	s.mu.Lock()
	ln := s.listener
	s.mu.Unlock()
	if ln == nil {
		return errors.New("pgproto: Serve called before Listen")
	}

	go func() {
		<-ctx.Done()
		_ = s.Close()
	}()

	for {
		conn, err := ln.Accept()
		if err != nil {
			s.mu.Lock()
			closed := s.closed
			s.mu.Unlock()
			if closed {
				return nil
			}
			return fmt.Errorf("pgproto: accept: %w", err)
		}
		if err := s.authorize(conn); err != nil {
			s.logger.Warn("pgproto: rejected connection", "error", err)
			_ = conn.Close()
			continue
		}
		s.track(conn)
		go s.handleConn(ctx, conn)
	}
}

// authorize enforces the peer-UID check when configured.
func (s *Server) authorize(conn net.Conn) error {
	if s.cfg.AllowedUID < 0 {
		return nil
	}
	uc, ok := conn.(*net.UnixConn)
	if !ok {
		return errors.New("pgproto: non-unix connection")
	}
	uid, err := peerUID(uc)
	if err != nil {
		return fmt.Errorf("pgproto: peer credential check: %w", err)
	}
	if uid != s.cfg.AllowedUID {
		return fmt.Errorf("pgproto: peer uid %d not allowed (want %d)", uid, s.cfg.AllowedUID)
	}
	return nil
}

func (s *Server) track(conn net.Conn) {
	s.mu.Lock()
	s.conns[conn] = struct{}{}
	s.mu.Unlock()
}

func (s *Server) untrack(conn net.Conn) {
	s.mu.Lock()
	delete(s.conns, conn)
	s.mu.Unlock()
}

// connState holds per-connection routing and search-cursor state.
type connState struct {
	collection string
	cursor     []ResultMsg
	cursorPos  int
}

// handleConn drives one connection's request/reply loop.
func (s *Server) handleConn(ctx context.Context, conn net.Conn) {
	defer s.untrack(conn)
	defer func() { _ = conn.Close() }()

	st := &connState{}
	for {
		frame, err := ReadFrame(conn)
		if err != nil {
			if !errors.Is(err, io.EOF) {
				s.logger.Debug("pgproto: connection closed", "error", err)
			}
			return
		}
		stop, err := s.dispatch(ctx, conn, st, frame)
		if err != nil {
			_ = WriteFrame(conn, OpError, EncodeError(err.Error()))
			continue
		}
		if stop {
			return
		}
	}
}

// dispatch handles a single frame, writing the reply. It returns stop=true on
// SHUTDOWN.
func (s *Server) dispatch(ctx context.Context, conn net.Conn, st *connState, f Frame) (stop bool, err error) {
	switch f.Opcode {
	case OpBuildBegin:
		m, derr := DecodeBuildBegin(f.Payload)
		if derr != nil {
			return false, derr
		}
		st.collection = m.Collection
		if err := s.handler.BuildBegin(ctx, m); err != nil {
			return false, err
		}
		return false, WriteFrame(conn, OpAck, nil)

	case OpBuildVector, OpInsert:
		m, derr := DecodeVectorMsg(f.Payload)
		if derr != nil {
			return false, derr
		}
		if err := s.requireCollection(st); err != nil {
			return false, err
		}
		if err := s.handler.Insert(ctx, st.collection, m); err != nil {
			return false, err
		}
		return false, WriteFrame(conn, OpAck, nil)

	case OpDelete:
		m, derr := DecodeDeleteMsg(f.Payload)
		if derr != nil {
			return false, derr
		}
		if err := s.requireCollection(st); err != nil {
			return false, err
		}
		if err := s.handler.Delete(ctx, st.collection, m); err != nil {
			return false, err
		}
		return false, WriteFrame(conn, OpAck, nil)

	case OpBuildCommit:
		if err := s.requireCollection(st); err != nil {
			return false, err
		}
		if err := s.handler.Commit(ctx, st.collection); err != nil {
			return false, err
		}
		return false, WriteFrame(conn, OpAck, nil)

	case OpSearchBegin:
		m, derr := DecodeSearchBegin(f.Payload)
		if derr != nil {
			return false, derr
		}
		st.collection = m.Collection
		results, serr := s.handler.Search(ctx, m)
		if serr != nil {
			return false, serr
		}
		st.cursor = results
		st.cursorPos = 0
		return false, WriteFrame(conn, OpAck, nil)

	case OpSearchNext:
		var row ResultMsg
		if st.cursorPos >= len(st.cursor) {
			row = ResultMsg{Done: true}
		} else {
			row = st.cursor[st.cursorPos]
			st.cursorPos++
		}
		return false, WriteFrame(conn, OpResult, row.Encode())

	case OpSearchEnd:
		st.cursor = nil
		st.cursorPos = 0
		return false, WriteFrame(conn, OpAck, nil)

	case OpStats:
		if err := s.requireCollection(st); err != nil {
			return false, err
		}
		reply, serr := s.handler.Stats(ctx, st.collection)
		if serr != nil {
			return false, serr
		}
		return false, WriteFrame(conn, OpStats, reply.Encode())

	case OpShutdown:
		_ = WriteFrame(conn, OpAck, nil)
		return true, nil

	default:
		return false, fmt.Errorf("pgproto: unexpected opcode %s", f.Opcode)
	}
}

func (s *Server) requireCollection(st *connState) error {
	if st.collection == "" {
		return errors.New("pgproto: no active collection (send BUILD_BEGIN or SEARCH_BEGIN first)")
	}
	return nil
}

// Addr returns the socket path the server is listening on.
func (s *Server) Addr() string { return s.cfg.SocketPath }

// Close stops accepting connections, closes open connections, and removes the
// socket file.
func (s *Server) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	ln := s.listener
	conns := make([]net.Conn, 0, len(s.conns))
	for c := range s.conns {
		conns = append(conns, c)
	}
	s.mu.Unlock()

	var firstErr error
	if ln != nil {
		if err := ln.Close(); err != nil {
			firstErr = err
		}
	}
	for _, c := range conns {
		_ = c.Close()
	}
	_ = os.Remove(s.cfg.SocketPath)
	return firstErr
}
