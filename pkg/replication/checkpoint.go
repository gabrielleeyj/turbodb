package replication

import (
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"os"
	"path/filepath"
)

// Checkpoint persists the last-committed source LSN so a restarted sync
// resumes without loss or duplication.
type Checkpoint interface {
	// Load returns the stored LSN, or (0, nil) if no checkpoint exists yet.
	Load() (uint64, error)
	// Save durably records the LSN.
	Save(lsn uint64) error
}

// checkpointFileSize is 8 bytes of little-endian LSN + 4 bytes of CRC32C.
const checkpointFileSize = 12

var crcCheckpoint = crc32.MakeTable(crc32.Castagnoli)

// FileCheckpoint stores the LSN in a single small file, written atomically
// (tmp + fsync + rename) and integrity-checked with CRC32C.
type FileCheckpoint struct {
	path string
}

// NewFileCheckpoint creates a checkpoint store at path. The parent directory
// must exist.
func NewFileCheckpoint(path string) (*FileCheckpoint, error) {
	if path == "" {
		return nil, fmt.Errorf("replication: checkpoint path is required")
	}
	if _, err := os.Stat(filepath.Dir(path)); err != nil {
		return nil, fmt.Errorf("replication: checkpoint dir: %w", err)
	}
	return &FileCheckpoint{path: path}, nil
}

// Load reads the stored LSN. A missing file returns (0, nil); a corrupt or
// truncated file returns an error so the operator can decide how to recover.
func (c *FileCheckpoint) Load() (uint64, error) {
	data, err := os.ReadFile(c.path) // #nosec G304 -- operator-configured checkpoint path
	if errors.Is(err, os.ErrNotExist) {
		return 0, nil
	}
	if err != nil {
		return 0, fmt.Errorf("replication: read checkpoint %s: %w", c.path, err)
	}
	if len(data) != checkpointFileSize {
		return 0, fmt.Errorf("replication: checkpoint %s: unexpected size %d", c.path, len(data))
	}
	lsn := binary.LittleEndian.Uint64(data[:8])
	stored := binary.LittleEndian.Uint32(data[8:12])
	if computed := crc32.Checksum(data[:8], crcCheckpoint); stored != computed {
		return 0, fmt.Errorf("replication: checkpoint %s: crc mismatch (stored=%x computed=%x)",
			c.path, stored, computed)
	}
	return lsn, nil
}

// Save atomically writes the LSN: write to a temp file in the same directory,
// fsync, then rename over the checkpoint path.
func (c *FileCheckpoint) Save(lsn uint64) error {
	var buf [checkpointFileSize]byte
	binary.LittleEndian.PutUint64(buf[:8], lsn)
	binary.LittleEndian.PutUint32(buf[8:12], crc32.Checksum(buf[:8], crcCheckpoint))

	tmp := c.path + ".tmp"
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600) // #nosec G304 -- operator-configured checkpoint path
	if err != nil {
		return fmt.Errorf("replication: create checkpoint tmp: %w", err)
	}
	if _, err := f.Write(buf[:]); err != nil {
		_ = f.Close()
		return fmt.Errorf("replication: write checkpoint: %w", err)
	}
	if err := f.Sync(); err != nil {
		_ = f.Close()
		return fmt.Errorf("replication: sync checkpoint: %w", err)
	}
	if err := f.Close(); err != nil {
		return fmt.Errorf("replication: close checkpoint: %w", err)
	}
	if err := os.Rename(tmp, c.path); err != nil {
		return fmt.Errorf("replication: rename checkpoint: %w", err)
	}
	return nil
}
