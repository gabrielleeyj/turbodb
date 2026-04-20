# Operations Guide

## Resource Sizing

| Component        | GPU           | CPU      | RAM      | Disk       |
| ---------------- | ------------- | -------- | -------- | ---------- |
| turbodb-engine   | A100 40GB min | 16 cores | 128 GB   | 1 TB+ NVMe |
| turbodb-sync     | None          | 2 cores  | 4 GB     | Small      |
| PostgreSQL + ext | None          | Standard | Standard | Standard   |

## Configuration

Engine configuration lives at `/etc/turbodb/engine.yaml` (bare metal) or in a
ConfigMap (Kubernetes).

## Monitoring

- Prometheus metrics at `/metrics` on the admin HTTP port (default 8080).
- OpenTelemetry traces for search request flow.
- Structured JSON logs via `log/slog`.

## Backup & Recovery

- WAL-based recovery: replay from last checkpoint on crash.
- Segment files are immutable once sealed; back up the segment directory.
- PostgreSQL-side: standard pg_basebackup + WAL archiving.
