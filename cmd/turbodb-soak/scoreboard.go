package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sync"
	"time"
)

// scoreboard accumulates the soak run's results. It is written as JSON at
// the end of the run and printed after every cycle.
type scoreboard struct {
	mu sync.Mutex

	StartedAt           time.Time      `json:"started_at"`
	Faults              map[string]int `json:"faults"`
	VerificationsPassed int            `json:"verifications_passed"`
	Violations          int            `json:"violations"`
	ViolationLog        []string       `json:"violation_log,omitempty"`
	SupervisedRestarts  int            `json:"supervised_restarts"`
	MaxCatchup          time.Duration  `json:"max_catchup_ns"`
	LastCatchup         time.Duration  `json:"last_catchup_ns"`
}

func newScoreboard() *scoreboard {
	return &scoreboard{StartedAt: time.Now(), Faults: make(map[string]int)}
}

func (b *scoreboard) violation(format string, args ...any) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.Violations++
	msg := fmt.Sprintf(format, args...)
	b.ViolationLog = append(b.ViolationLog, time.Now().Format(time.RFC3339)+" "+msg)
	fmt.Fprintf(os.Stderr, "VIOLATION: %s\n", msg)
}

func (b *scoreboard) fault(name string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.Faults[name]++
}

func (b *scoreboard) supervisedRestart() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.SupervisedRestarts++
}

func (b *scoreboard) verificationPassed() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.VerificationsPassed++
}

func (b *scoreboard) violations() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.Violations
}

func (b *scoreboard) verifications() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.VerificationsPassed
}

func (b *scoreboard) recordCatchup(d time.Duration) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.LastCatchup = d
	if d > b.MaxCatchup {
		b.MaxCatchup = d
	}
}

func (b *scoreboard) print(w io.Writer) {
	b.mu.Lock()
	defer b.mu.Unlock()
	fmt.Fprintf(w, "[soak %s] faults=%v verified=%d violations=%d sync_restarts=%d max_catchup=%s\n",
		time.Since(b.StartedAt).Round(time.Second), b.Faults,
		b.VerificationsPassed, b.Violations, b.SupervisedRestarts,
		b.MaxCatchup.Round(time.Millisecond))
}

func (b *scoreboard) write(path string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	data, err := json.MarshalIndent(b, "", "  ")
	if err != nil {
		return
	}
	_ = os.WriteFile(path, data, 0o600)
}
