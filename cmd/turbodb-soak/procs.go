package main

import (
	"log/slog"
	"os"
	"os/exec"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

// managedProc supervises one binary the way systemd would: it restarts the
// process on every exit until stop() is called. Exits that were not caused
// by an injected fault are counted as violations unless exitsAreExpected
// (sync's circuit breaker exits by design).
type managedProc struct {
	name    string
	logger  *slog.Logger
	board   *scoreboard
	bin     string
	args    []string
	logPath string

	// restartDelay is the fixed supervisor pause before a restart.
	restartDelay time.Duration
	// nextRestartDelay, when set, overrides restartDelay once (crash
	// faults use it to simulate an outage window).
	nextRestartDelay atomic.Int64
	// expectedKills counts injected kills that have not yet been matched
	// with an exit, so they are not scored as unexpected deaths.
	expectedKills atomic.Int64
	// exitsAreExpected marks processes whose self-exit is part of the
	// design (supervision model) rather than a violation.
	exitsAreExpected bool

	mu      sync.Mutex
	cmd     *exec.Cmd
	stopped bool
	done    chan struct{}
}

func newManagedProc(name string, logger *slog.Logger, board *scoreboard,
	restartDelay time.Duration, bin string, args ...string) *managedProc {
	return &managedProc{
		name: name, logger: logger, board: board,
		restartDelay: restartDelay, bin: bin, args: args,
		done: make(chan struct{}),
	}
}

func (p *managedProc) logTo(path string) { p.logPath = path }

// start launches the process and the supervision loop.
func (p *managedProc) start() {
	go p.superviseLoop()
}

func (p *managedProc) superviseLoop() {
	defer close(p.done)
	for {
		p.mu.Lock()
		if p.stopped {
			p.mu.Unlock()
			return
		}
		cmd := exec.Command(p.bin, p.args...) // #nosec G204 -- harness-configured binary paths
		var logFile *os.File
		if p.logPath != "" {
			if f, err := os.OpenFile(p.logPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600); err == nil { // #nosec G304 -- harness workdir
				cmd.Stdout, cmd.Stderr = f, f
				logFile = f
			}
		}
		if err := cmd.Start(); err != nil {
			p.mu.Unlock()
			p.logger.Error("proc start failed", "proc", p.name, "err", err)
			p.board.violation("%s failed to start: %v", p.name, err)
			return
		}
		p.cmd = cmd
		p.mu.Unlock()
		p.logger.Info("proc started", "proc", p.name, "pid", cmd.Process.Pid)

		err := cmd.Wait()
		if logFile != nil {
			_ = logFile.Close()
		}

		p.mu.Lock()
		stopped := p.stopped
		p.mu.Unlock()
		if stopped {
			return
		}

		if p.expectedKills.Load() > 0 {
			p.expectedKills.Add(-1)
		} else if p.exitsAreExpected {
			p.board.supervisedRestart()
			p.logger.Info("proc exited; supervisor restarting", "proc", p.name, "err", err)
		} else {
			p.board.violation("%s exited unexpectedly: %v", p.name, err)
			p.logger.Error("proc exited unexpectedly", "proc", p.name, "err", err)
		}

		delay := p.restartDelay
		if d := p.nextRestartDelay.Swap(0); d > 0 {
			delay = time.Duration(d)
		}
		if delay > 0 {
			time.Sleep(delay)
		}
	}
}

// kill SIGKILLs the process as an injected fault; the supervisor restarts it
// after outage (0 = immediately with the default delay).
func (p *managedProc) kill(outage time.Duration) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.cmd == nil || p.cmd.Process == nil {
		return
	}
	p.expectedKills.Add(1)
	if outage > 0 {
		p.nextRestartDelay.Store(int64(outage))
	}
	_ = p.cmd.Process.Kill()
}

// stall suspends the process with SIGSTOP for d, then resumes it. This
// stands in for a network partition: peers see an unresponsive endpoint
// until timeouts fire.
func (p *managedProc) stall(d time.Duration) {
	p.mu.Lock()
	proc := p.cmd.Process
	p.mu.Unlock()
	if proc == nil {
		return
	}
	_ = proc.Signal(syscall.SIGSTOP)
	time.Sleep(d)
	_ = proc.Signal(syscall.SIGCONT)
}

// stop terminates the process permanently.
func (p *managedProc) stop() {
	p.mu.Lock()
	p.stopped = true
	if p.cmd != nil && p.cmd.Process != nil {
		_ = p.cmd.Process.Kill()
	}
	p.mu.Unlock()
	select {
	case <-p.done:
	case <-time.After(10 * time.Second):
		p.logger.Error("proc supervisor did not stop in time", "proc", p.name)
	}
}
