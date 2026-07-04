//go:build darwin

package pgproto

import (
	"fmt"
	"net"

	"golang.org/x/sys/unix"
)

// peerUID returns the UID of the connected peer via LOCAL_PEERCRED.
func peerUID(conn *net.UnixConn) (int, error) {
	raw, err := conn.SyscallConn()
	if err != nil {
		return 0, err
	}
	var uid int
	var sockErr error
	if cerr := raw.Control(func(fd uintptr) {
		xucred, e := unix.GetsockoptXucred(int(fd), unix.SOL_LOCAL, unix.LOCAL_PEERCRED) // #nosec G115 -- fd fits in int on all supported platforms
		if e != nil {
			sockErr = e
			return
		}
		uid = int(xucred.Uid)
	}); cerr != nil {
		return 0, cerr
	}
	if sockErr != nil {
		return 0, fmt.Errorf("getsockopt LOCAL_PEERCRED: %w", sockErr)
	}
	return uid, nil
}
