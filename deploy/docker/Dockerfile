# Multi-stage build for the TurboDB engine and sync daemons (CPU-only).
# Build from the repository root:
#   docker build -f deploy/docker/Dockerfile --target engine -t turbodb-engine .
#   docker build -f deploy/docker/Dockerfile --target sync   -t turbodb-sync .

FROM golang:1.26-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags "-s -w" -o /out/ \
    ./cmd/turbodb-engine ./cmd/turbodb-sync ./cmd/turbodb-ctl

# --- Engine runtime ---
FROM alpine:3.21 AS engine
RUN adduser -D -u 10001 turbodb \
    && mkdir -p /var/lib/turbodb \
    && chown turbodb:turbodb /var/lib/turbodb
COPY --from=build /out/turbodb-engine /out/turbodb-ctl /usr/local/bin/
USER turbodb
VOLUME /var/lib/turbodb
# gRPC, admin HTTP/JSON, Prometheus metrics
EXPOSE 7080 8080 9090
ENTRYPOINT ["turbodb-engine"]
CMD ["--listen", ":7080", "--admin-listen", ":8080", "--metrics-listen", ":9090", \
     "--data-dir", "/var/lib/turbodb", "--log-format", "json", "--wal-fsync", "group"]

# --- Sync runtime ---
FROM alpine:3.21 AS sync
RUN adduser -D -u 10002 turbodb-sync \
    && mkdir -p /var/lib/turbodb-sync \
    && chown turbodb-sync:turbodb-sync /var/lib/turbodb-sync
COPY --from=build /out/turbodb-sync /out/turbodb-ctl /usr/local/bin/
USER turbodb-sync
VOLUME /var/lib/turbodb-sync
ENTRYPOINT ["turbodb-sync"]
CMD ["run", "--config", "/etc/turbodb/sync.yaml", \
     "--checkpoint", "/var/lib/turbodb-sync/sync.ckpt"]
