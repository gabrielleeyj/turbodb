package wal

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
)

// IterateOptions configures WAL iteration during recovery.
type IterateOptions struct {
	// FromLSN, when non-zero, skips records with LSN < FromLSN.
	FromLSN uint64
	// StopOnCorruption, when true, returns the first CRC error encountered.
	// Default (false) treats trailing corruption as end-of-stream — appropriate
	// for crash recovery where the tail of the last file may be torn.
	StopOnCorruption bool
}

// VisitFunc is invoked for each record during Iterate. Return a non-nil error
// to stop iteration; the same error is returned from Iterate.
type VisitFunc func(rec Record) error

// ErrStopIteration is a sentinel used by callers to halt Iterate without
// signalling an error. Iterate returns nil when a visitor returns this value.
var ErrStopIteration = errors.New("wal: stop iteration")

// Iterate walks every record in the WAL directory in LSN order and invokes fn
// for each one whose LSN >= opts.FromLSN.
//
// Iterate opens files independently of any active writer, so it is safe to
// call concurrently with Append. The active file is read up to its current
// committed length.
func Iterate(dir string, opts IterateOptions, fn VisitFunc) error {
	files, err := listWALFiles(dir)
	if err != nil {
		return fmt.Errorf("wal: iterate scan: %w", err)
	}

	for _, file := range files {
		stop, err := iterateFile(file.path, opts, fn)
		if err != nil {
			return err
		}
		if stop {
			return nil
		}
	}
	return nil
}

// iterateFile reads records from a single file. Returns (stop, err) where stop
// indicates the visitor signalled ErrStopIteration.
func iterateFile(path string, opts IterateOptions, fn VisitFunc) (bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return false, fmt.Errorf("wal: open %s: %w", path, err)
	}
	defer f.Close()

	r := bufio.NewReader(f)
	for {
		rec, _, err := readRecord(r)
		if err != nil {
			if errors.Is(err, io.EOF) {
				return false, nil
			}
			if errors.Is(err, ErrCorruptRecord) {
				if opts.StopOnCorruption {
					return false, fmt.Errorf("wal: %s: %w", path, err)
				}
				return false, nil
			}
			return false, fmt.Errorf("wal: read %s: %w", path, err)
		}

		if rec.LSN < opts.FromLSN {
			continue
		}

		if err := fn(rec); err != nil {
			if errors.Is(err, ErrStopIteration) {
				return true, nil
			}
			return false, err
		}
	}
}

// LastCheckpointLSN scans the WAL for the most recent OpCheckpoint record and
// returns the durable LSN it carries. Returns 0 if no checkpoint is found.
func LastCheckpointLSN(dir string) (uint64, error) {
	files, err := listWALFiles(dir)
	if err != nil {
		return 0, fmt.Errorf("wal: checkpoint scan: %w", err)
	}

	var latest uint64
	// Walk newest-to-oldest for early termination.
	for i := len(files) - 1; i >= 0; i-- {
		lsn, found, err := scanFileForCheckpoint(files[i].path)
		if err != nil {
			return 0, err
		}
		if found {
			if lsn > latest {
				latest = lsn
			}
			return latest, nil
		}
	}
	return latest, nil
}

// scanFileForCheckpoint returns the highest checkpoint LSN within the file.
func scanFileForCheckpoint(path string) (uint64, bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, false, err
	}
	defer f.Close()

	r := bufio.NewReader(f)
	var bestLSN uint64
	var found bool
	for {
		rec, _, err := readRecord(r)
		if err != nil {
			break
		}
		if rec.Type != OpCheckpoint {
			continue
		}
		cp, derr := DecodeCheckpoint(rec.Payload)
		if derr != nil {
			continue
		}
		if cp.LSN >= bestLSN {
			bestLSN = cp.LSN
			found = true
		}
	}
	return bestLSN, found, nil
}
