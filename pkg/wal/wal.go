package wal

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// FsyncPolicy controls when WAL writes are flushed to disk.
type FsyncPolicy int

const (
	// FsyncEveryWrite calls fsync after every Append. Strongest durability,
	// lowest throughput.
	FsyncEveryWrite FsyncPolicy = iota
	// FsyncGroupCommit batches fsyncs at a fixed interval (GroupCommitInterval).
	// Append blocks until the next group fsync completes.
	FsyncGroupCommit
)

// String returns a human-readable policy name.
func (p FsyncPolicy) String() string {
	switch p {
	case FsyncEveryWrite:
		return "every_write"
	case FsyncGroupCommit:
		return "group_commit"
	default:
		return fmt.Sprintf("unknown(%d)", int(p))
	}
}

// Config controls WAL behavior.
type Config struct {
	// Dir is the directory where WAL files are stored. Created if missing.
	Dir string
	// MaxFileBytes is the file rotation threshold. Default: 100 MiB.
	MaxFileBytes int64
	// FsyncPolicy controls fsync timing. Default: FsyncEveryWrite.
	FsyncPolicy FsyncPolicy
	// GroupCommitInterval is the fsync cadence when FsyncGroupCommit is set.
	// Default: 10ms.
	GroupCommitInterval time.Duration
	// Logger receives operational logs. Defaults to slog.Default().
	Logger *slog.Logger
}

// Default values.
const (
	DefaultMaxFileBytes        int64         = 100 * 1024 * 1024
	DefaultGroupCommitInterval time.Duration = 10 * time.Millisecond
	walFilePrefix                            = "wal-"
	walFileSuffix                            = ".log"
)

// WAL is an append-only write-ahead log split across rotated files.
// All exported methods are safe for concurrent use.
type WAL struct {
	dir          string
	maxFileBytes int64
	policy       FsyncPolicy
	commitEvery  time.Duration
	logger       *slog.Logger

	mu       sync.Mutex
	curFile  *os.File
	curWrite *bufio.Writer
	curBytes int64
	curIndex uint64 // current WAL file index (numeric suffix).

	// nextLSN is the LSN that will be assigned to the next Append.
	nextLSN atomic.Uint64

	// Group-commit coordination.
	groupCh   chan groupReq
	groupWg   sync.WaitGroup
	groupStop context.CancelFunc

	closed atomic.Bool
}

type groupReq struct {
	done chan error
}

// Open opens or creates a WAL in cfg.Dir. If the directory contains existing
// WAL files, the highest-LSN file is opened for append and nextLSN is set to
// last_lsn + 1.
func Open(cfg Config) (*WAL, error) {
	if cfg.Dir == "" {
		return nil, fmt.Errorf("wal: Dir must not be empty")
	}
	if cfg.MaxFileBytes <= 0 {
		cfg.MaxFileBytes = DefaultMaxFileBytes
	}
	if cfg.GroupCommitInterval <= 0 {
		cfg.GroupCommitInterval = DefaultGroupCommitInterval
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}

	if err := os.MkdirAll(cfg.Dir, 0o755); err != nil {
		return nil, fmt.Errorf("wal: create dir %q: %w", cfg.Dir, err)
	}

	w := &WAL{
		dir:          cfg.Dir,
		maxFileBytes: cfg.MaxFileBytes,
		policy:       cfg.FsyncPolicy,
		commitEvery:  cfg.GroupCommitInterval,
		logger:       cfg.Logger,
	}

	files, err := listWALFiles(cfg.Dir)
	if err != nil {
		return nil, fmt.Errorf("wal: scan dir: %w", err)
	}

	var startIndex uint64
	var startLSN uint64
	if len(files) == 0 {
		startIndex = 1
		startLSN = 1
	} else {
		// Determine the next LSN by scanning the highest-index file.
		last := files[len(files)-1]
		startIndex = last.index
		lsn, err := scanLastLSN(last.path)
		if err != nil {
			return nil, fmt.Errorf("wal: scan last LSN in %s: %w", last.path, err)
		}
		startLSN = lsn + 1
	}

	w.curIndex = startIndex
	w.nextLSN.Store(startLSN)

	if err := w.openCurrentFile(); err != nil {
		return nil, err
	}

	if w.policy == FsyncGroupCommit {
		ctx, cancel := context.WithCancel(context.Background())
		w.groupStop = cancel
		w.groupCh = make(chan groupReq, 256)
		w.groupWg.Add(1)
		go w.groupCommitLoop(ctx)
	}

	return w, nil
}

// NextLSN returns the LSN that will be assigned to the next successful Append.
func (w *WAL) NextLSN() uint64 {
	return w.nextLSN.Load()
}

// Append writes a record with the given type and payload, returning the
// assigned LSN. Append is durable on return when FsyncEveryWrite is set, or
// when the next group-commit fsync completes if FsyncGroupCommit is set.
func (w *WAL) Append(typ RecordType, payload []byte) (uint64, error) {
	if w.closed.Load() {
		return 0, fmt.Errorf("wal: closed")
	}

	w.mu.Lock()
	lsn := w.nextLSN.Add(1) - 1
	rec := Record{LSN: lsn, Type: typ, Payload: payload}

	size := EncodedSize(len(payload))
	if w.curBytes > 0 && w.curBytes+int64(size) > w.maxFileBytes {
		if err := w.rotateLocked(); err != nil {
			w.mu.Unlock()
			return 0, err
		}
	}

	n, err := writeRecord(w.curWrite, rec)
	if err != nil {
		w.mu.Unlock()
		return 0, err
	}
	w.curBytes += int64(n)

	if w.policy == FsyncEveryWrite {
		if err := w.flushAndSyncLocked(); err != nil {
			w.mu.Unlock()
			return 0, fmt.Errorf("wal: fsync: %w", err)
		}
		w.mu.Unlock()
		return lsn, nil
	}

	// Group commit: enqueue and wait.
	w.mu.Unlock()
	req := groupReq{done: make(chan error, 1)}
	select {
	case w.groupCh <- req:
	default:
		// Channel full — fall back to synchronous fsync to avoid blocking forever.
		w.mu.Lock()
		err := w.flushAndSyncLocked()
		w.mu.Unlock()
		if err != nil {
			return 0, fmt.Errorf("wal: fsync (fallback): %w", err)
		}
		return lsn, nil
	}
	if err := <-req.done; err != nil {
		return 0, err
	}
	return lsn, nil
}

// Sync forces a buffer flush and fsync of the current file.
func (w *WAL) Sync() error {
	if w.closed.Load() {
		return fmt.Errorf("wal: closed")
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.flushAndSyncLocked()
}

// Close flushes pending writes, syncs to disk, and releases resources.
func (w *WAL) Close() error {
	if !w.closed.CompareAndSwap(false, true) {
		return nil
	}

	if w.groupStop != nil {
		w.groupStop()
		w.groupWg.Wait()
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	if w.curFile == nil {
		return nil
	}
	if err := w.flushAndSyncLocked(); err != nil {
		w.curFile.Close()
		w.curFile = nil
		return err
	}
	err := w.curFile.Close()
	w.curFile = nil
	w.curWrite = nil
	return err
}

// Truncate deletes WAL files whose records all precede beforeLSN. Files whose
// first LSN is < beforeLSN AND whose last LSN is < beforeLSN are removed.
// The currently-active file is never deleted.
func (w *WAL) Truncate(beforeLSN uint64) error {
	if w.closed.Load() {
		return fmt.Errorf("wal: closed")
	}
	w.mu.Lock()
	currentIndex := w.curIndex
	w.mu.Unlock()

	files, err := listWALFiles(w.dir)
	if err != nil {
		return fmt.Errorf("wal: truncate scan: %w", err)
	}

	for _, f := range files {
		if f.index == currentIndex {
			continue
		}
		lastLSN, err := scanLastLSN(f.path)
		if err != nil {
			w.logger.Warn("wal: truncate scan failed; skipping", "file", f.path, "error", err)
			continue
		}
		if lastLSN < beforeLSN {
			if err := os.Remove(f.path); err != nil {
				return fmt.Errorf("wal: remove %s: %w", f.path, err)
			}
			w.logger.Info("wal: truncated file", "file", f.path, "last_lsn", lastLSN)
		}
	}
	return nil
}

// Files returns the list of WAL file paths in LSN order (oldest first).
// Useful for iteration / recovery.
func (w *WAL) Files() ([]string, error) {
	files, err := listWALFiles(w.dir)
	if err != nil {
		return nil, err
	}
	paths := make([]string, len(files))
	for i, f := range files {
		paths[i] = f.path
	}
	return paths, nil
}

// Dir returns the directory the WAL was opened in.
func (w *WAL) Dir() string { return w.dir }

// openCurrentFile opens (or creates) the file at w.curIndex for append.
func (w *WAL) openCurrentFile() error {
	path := walFilePath(w.dir, w.curIndex)
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR|os.O_APPEND, 0o644)
	if err != nil {
		return fmt.Errorf("wal: open %s: %w", path, err)
	}
	info, err := f.Stat()
	if err != nil {
		f.Close()
		return fmt.Errorf("wal: stat %s: %w", path, err)
	}
	w.curFile = f
	w.curWrite = bufio.NewWriterSize(f, 64*1024)
	w.curBytes = info.Size()
	return nil
}

// rotateLocked closes the current file and opens a new one. Caller holds w.mu.
func (w *WAL) rotateLocked() error {
	if err := w.flushAndSyncLocked(); err != nil {
		return fmt.Errorf("wal: rotate sync: %w", err)
	}
	if err := w.curFile.Close(); err != nil {
		return fmt.Errorf("wal: rotate close: %w", err)
	}
	w.curIndex++
	if err := w.openCurrentFile(); err != nil {
		return fmt.Errorf("wal: rotate open: %w", err)
	}
	w.logger.Info("wal: rotated to new file", "index", w.curIndex, "path", w.curFile.Name())
	return nil
}

// flushAndSyncLocked flushes the buffered writer and calls Sync. Caller holds w.mu.
func (w *WAL) flushAndSyncLocked() error {
	if w.curWrite != nil {
		if err := w.curWrite.Flush(); err != nil {
			return err
		}
	}
	if w.curFile != nil {
		return w.curFile.Sync()
	}
	return nil
}

// groupCommitLoop fsyncs at commitEvery intervals and notifies waiters.
func (w *WAL) groupCommitLoop(ctx context.Context) {
	defer w.groupWg.Done()
	ticker := time.NewTicker(w.commitEvery)
	defer ticker.Stop()

	pending := make([]groupReq, 0, 64)
	flush := func() {
		if len(pending) == 0 {
			return
		}
		w.mu.Lock()
		err := w.flushAndSyncLocked()
		w.mu.Unlock()
		for _, r := range pending {
			r.done <- err
		}
		pending = pending[:0]
	}

	for {
		select {
		case <-ctx.Done():
			// Drain any remaining requests before exit.
			for {
				select {
				case req := <-w.groupCh:
					pending = append(pending, req)
				default:
					flush()
					return
				}
			}
		case req := <-w.groupCh:
			pending = append(pending, req)
		case <-ticker.C:
			flush()
		}
	}
}

// walFile describes a discovered WAL file on disk.
type walFile struct {
	index uint64
	path  string
}

// listWALFiles returns WAL files in dir sorted by ascending index.
func listWALFiles(dir string) ([]walFile, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}
	var files []walFile
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if !strings.HasPrefix(name, walFilePrefix) || !strings.HasSuffix(name, walFileSuffix) {
			continue
		}
		idxStr := strings.TrimSuffix(strings.TrimPrefix(name, walFilePrefix), walFileSuffix)
		idx, err := strconv.ParseUint(idxStr, 10, 64)
		if err != nil {
			continue
		}
		files = append(files, walFile{index: idx, path: filepath.Join(dir, name)})
	}
	sort.Slice(files, func(i, j int) bool { return files[i].index < files[j].index })
	return files, nil
}

// walFilePath returns the canonical path for a WAL file with the given index.
func walFilePath(dir string, index uint64) string {
	return filepath.Join(dir, fmt.Sprintf("%s%010d%s", walFilePrefix, index, walFileSuffix))
}

// scanLastLSN reads through a WAL file and returns the LSN of its last
// well-formed record. Returns 0 if the file is empty or contains no readable
// records. Trailing corruption is treated as a tail-truncation point.
func scanLastLSN(path string) (uint64, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	r := bufio.NewReader(f)
	var lastLSN uint64
	for {
		rec, _, err := readRecord(r)
		if err != nil {
			// EOF or trailing corruption — stop scanning.
			return lastLSN, nil
		}
		lastLSN = rec.LSN
	}
}
