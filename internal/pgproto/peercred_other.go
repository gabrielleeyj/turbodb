//go:build !linux && !darwin

package pgproto

import (
	"errors"
	"net"
)

// peerUID is unsupported on this platform; the server should be configured with
// AllowedUID < 0 to skip the check.
func peerUID(conn *net.UnixConn) (int, error) {
	return 0, errors.New("pgproto: peer credential check unsupported on this platform")
}
