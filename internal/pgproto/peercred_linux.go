//go:build linux

package pgproto

import (
	"fmt"
	"net"

	"golang.org/x/sys/unix"
)

// peerUID returns the effective UID of the connected peer via SO_PEERCRED.
func peerUID(conn *net.UnixConn) (int, error) {
	raw, err := conn.SyscallConn()
	if err != nil {
		return 0, err
	}
	var uid int
	var sockErr error
	if cerr := raw.Control(func(fd uintptr) {
		cred, e := unix.GetsockoptUcred(int(fd), unix.SOL_SOCKET, unix.SO_PEERCRED)
		if e != nil {
			sockErr = e
			return
		}
		uid = int(cred.Uid)
	}); cerr != nil {
		return 0, cerr
	}
	if sockErr != nil {
		return 0, fmt.Errorf("getsockopt SO_PEERCRED: %w", sockErr)
	}
	return uid, nil
}
