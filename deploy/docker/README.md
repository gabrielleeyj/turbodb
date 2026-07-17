# TurboDB Docker Compose Stack

Production-testing deployment: PostgreSQL (logical replication source),
`turbodb-engine`, and `turbodb-sync`, built CPU-only from this repository.

## Bring-up

```sh
cd deploy/docker
cp .env.example .env            # set a real POSTGRES_PASSWORD
cp sync.yaml.example sync.yaml  # adjust table mappings if needed
docker compose up -d --build
```

Create the target collection once the engine is healthy (dimension must match
the `vector(N)` column and your sync mapping):

```sh
docker compose exec turbodb-engine turbodb-ctl collection create \
  --engine 127.0.0.1:7080 --name docs --dim 8 --bits 4
```

Insert rows into Postgres and they replicate to the engine:

```sh
docker compose exec postgres psql -U postgres -c \
  "INSERT INTO docs (doc_id, embedding) VALUES ('a', '[1,0,0,0,0,0,0,0]')"
docker compose exec turbodb-engine turbodb-ctl collection describe docs \
  --engine 127.0.0.1:7080
```

## Endpoints (published on loopback only)

| Port | Service | Notes |
|------|---------|-------|
| 7080 | Engine gRPC | No TLS/auth — private networks only |
| 8080 | Admin HTTP/JSON (`/healthz`, `/readyz`, `/api/v1/collections`) | Mutating routes require mTLS when enabled |
| 9090 | Prometheus metrics (`/metrics`) | |
| 5435 | PostgreSQL (host side; 5432 in-network) | |

## Enabling admin mTLS

The admin API has no authentication by default; it is bound to loopback for
that reason. To expose it, mount certificates and add the TLS flags to the
engine command:

```yaml
  turbodb-engine:
    command:
      - --listen=:7080
      - --admin-listen=:8080
      - --metrics-listen=:9090
      - --data-dir=/var/lib/turbodb
      - --log-format=json
      - --wal-fsync=group
      - --admin-tls-cert=/etc/turbodb/tls/server.crt
      - --admin-tls-key=/etc/turbodb/tls/server.key
      - --admin-tls-ca=/etc/turbodb/tls/ca.crt
    volumes:
      - enginedata:/var/lib/turbodb
      - ./tls:/etc/turbodb/tls:ro
```

Setting `--admin-tls-ca` enables mutual TLS (`RequireAndVerifyClientCert`)
and gates mutating admin routes on a verified client certificate.

## Supervision notes

- `turbodb-sync` exits deliberately when the engine is unreachable (circuit
  breaker); `restart: always` is the required supervision contract, mirroring
  the soak harness.
- The sync DSN is passed via the `TURBODB_PG_DSN` environment variable, never
  on the command line, so it does not appear in `ps` output.
- Reconciliation (drift repair) can be run ad hoc:

```sh
docker compose run --rm turbodb-sync reconcile \
  --config /etc/turbodb/sync.yaml --engine turbodb-engine:7080 --repair
```
