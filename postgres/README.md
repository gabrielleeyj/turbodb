# pg_turboquant

A PostgreSQL index access method that delegates GPU-accelerated TurboQuant
vector indexing to the `turbodb-engine` daemon over a Unix-socket IPC channel
(the PG-Strom pattern). It wraps pgvector's `vector` type rather than replacing
it.

## Components

| File | Purpose |
| --- | --- |
| `pg_turboquant.c` | Access-method handler (`IndexAmRoutine`), GUCs, reloptions |
| `turbodb_ipc.{c,h}` | Self-contained C client for the engine IPC protocol |
| `turbodb_ipc_probe.c` | Standalone harness that exercises the IPC client |
| `pg_turboquant.control`, `pg_turboquant--0.1.sql` | Extension metadata + DDL |
| `Makefile` | PGXS build |

The C IPC client is the exact counterpart of the Go `internal/pgproto`
implementation; the two are validated against each other by
`TestCClientWireCompatibility` in `internal/pgipc`, which compiles this client
and runs it against the Go server.

## Build

```sh
make                 # builds the standalone IPC self-test harness
make all             # builds the extension shared library (needs pg_config)
make install         # installs into the PostgreSQL lib/extension dirs
PG_CONFIG=/path/to/pg_config make all   # target a specific server
```

Developed and compiled clean against **PostgreSQL 18** (Postgres.app); the
handler uses `#if`-free code that should also build on 16/17. The DDL depends on
the `vector` type, so **pgvector must be installed** for the target server
before `CREATE EXTENSION`.

## Configure & use

```sql
CREATE EXTENSION pg_turboquant;          -- requires pgvector

-- point the extension at the engine's IPC socket (postgresql.conf or SET)
SET pg_turboquant.engine_socket = '/var/run/turbodb/engine.sock';

CREATE INDEX ON items USING turboquant (embedding) WITH (bits = 4, oversearch_factor = 2);

SELECT id FROM items ORDER BY embedding <-> '[...]' LIMIT 10;
```

Run the engine with the IPC server enabled:

```sh
turbodb-engine --data-dir ./data --pg-socket /var/run/turbodb/engine.sock
```

## Status (Phase 5)

Implemented and **compile-verified locally**:

- Task 4.1 — extension skeleton (`.control`, DDL, PGXS Makefile, `PG_MODULE_MAGIC`, `_PG_init`).
- Task 4.2 — `IndexAmRoutine`: `ambuild` (heap scan → `BUILD_BEGIN`/`BUILD_VECTOR`/`BUILD_COMMIT`), `ambuildempty`, `aminsert`, `ambulkdelete`/`amvacuumcleanup` (stubs), `ambeginscan`/`amrescan`/`amgettuple`/`amendscan` (IPC search), `amoptions` (`bits`, `oversearch_factor`, `use_qjl`), `amcostestimate`, `amvalidate`.
- Task 4.3 — the binary IPC protocol (Go server in `internal/pgproto`, C client here), wire-compatibility tested across both implementations.
- Task 4.5 — `pg_turboquant_indexes` catalog view.

Deferred (require a running engine + pgvector-on-18 + GPU, i.e. integration
hardware not available on the dev box):

- Task 4.4 — Custom WAL Resource Manager registration for crash-consistent
  replay (the engine already has its own WAL; PG-side custom RM is the
  remaining piece).
- Full acceptance tests `TestPgTurboquantBuild/Query/CrashRecovery` (need a live
  cluster with pgvector and the GPU engine).
- `EXPLAIN` custom plan-node annotations.
- `SO_PEERCRED` is enforced on the **engine** side (`--pg-allowed-uid`); the C
  client relies on filesystem permissions of the socket.
