# TurboDB systemd Units

Single-host deployment of `turbodb-engine` and `turbodb-sync` under systemd.

## Install

```sh
# Binaries (build with CGO_ENABLED=0 go build ./cmd/... or copy from CI)
install -m 0755 turbodb-engine turbodb-sync turbodb-ctl /usr/local/bin/

# Users
useradd --system --home /var/lib/turbodb      --shell /usr/sbin/nologin turbodb
useradd --system --home /var/lib/turbodb-sync --shell /usr/sbin/nologin turbodb-sync

# Config
mkdir -p /etc/turbodb
install -m 0640 -o root -g turbodb      engine.env.example /etc/turbodb/engine.env
install -m 0640 -o root -g turbodb-sync sync.env.example   /etc/turbodb/sync.env
install -m 0640 -o root -g turbodb-sync sync.yaml          /etc/turbodb/sync.yaml

# Units
cp turbodb-engine.service turbodb-sync.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now turbodb-engine
systemctl enable --now turbodb-sync
```

`StateDirectory=` creates `/var/lib/turbodb` and `/var/lib/turbodb-sync`
with correct ownership automatically; no manual `mkdir` needed.

## PostgreSQL prerequisites

- `wal_level = logical` in `postgresql.conf` (restart required).
- A publication covering the synced tables:
  `CREATE PUBLICATION turbodb_pub FOR TABLE public.docs;`
- The sync role needs `REPLICATION` and `SELECT` on the synced tables.

## Verify

```sh
systemctl status turbodb-engine turbodb-sync
curl -s http://127.0.0.1:8080/readyz          # engine readiness
turbodb-ctl collection list --engine 127.0.0.1:7080
turbodb-ctl sync status --checkpoint /var/lib/turbodb-sync/sync.ckpt
```

## Security notes

- The admin API (`127.0.0.1:8080`) has no authentication unless mTLS is
  enabled via `TURBODB_ENGINE_EXTRA_FLAGS`; keep it on loopback otherwise.
- Engine gRPC (`:7080`) has no TLS; firewall it to trusted networks.
- The DSN lives in `/etc/turbodb/sync.env` (0640), never on the command line.
- Units run unprivileged with `ProtectSystem=strict`; writable paths are
  limited to each service's state directory.
