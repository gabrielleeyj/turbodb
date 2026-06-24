# PGXS Makefile for pg_turboquant.
#
# Build:   make            (uses pg_config on PATH; override with PG_CONFIG=...)
# Install: make install
# Test:    make installcheck   (requires a running server + pgvector + engine)

MODULE_big = pg_turboquant
OBJS = pg_turboquant.o turbodb_ipc.o

EXTENSION = pg_turboquant
DATA = pg_turboquant--0.1.sql
PGFILEDESC = "pg_turboquant - GPU-accelerated TurboQuant vector index"

# Standalone unit check for the IPC client (no PostgreSQL needed).
ipc-selftest:
	$(CC) -std=c11 -Wall -Wextra -I. turbodb_ipc.c turbodb_ipc_probe.c -o /tmp/turbodb_ipc_probe
	@echo "built /tmp/turbodb_ipc_probe (run with a socket path to exercise the engine)"

PG_CONFIG ?= pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
