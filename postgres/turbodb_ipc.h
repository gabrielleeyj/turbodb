/*
 * turbodb_ipc.h — C client for the TurboDB engine IPC protocol.
 *
 * Mirrors internal/pgproto (Go): a length-prefixed binary framing over a
 * SOCK_STREAM Unix socket.
 *
 *   [4 bytes big-endian uint32 length L]   bytes that follow (opcode + payload)
 *   [2 bytes big-endian uint16 opcode]
 *   [2 bytes little-endian uint16 schema version]
 *   [payload, little-endian]
 *
 * This header is self-contained (libc + POSIX sockets only) so it can be
 * compiled and unit-tested independently of PostgreSQL.
 */
#ifndef TURBODB_IPC_H
#define TURBODB_IPC_H

#include <stdint.h>
#include <stddef.h>

/* Opcodes — must match internal/pgproto/protocol.go. */
typedef enum {
	TQ_OP_BUILD_BEGIN  = 1,
	TQ_OP_BUILD_VECTOR = 2,
	TQ_OP_BUILD_COMMIT = 3,
	TQ_OP_INSERT       = 4,
	TQ_OP_DELETE       = 5,
	TQ_OP_SEARCH_BEGIN = 6,
	TQ_OP_SEARCH_NEXT  = 7,
	TQ_OP_SEARCH_END   = 8,
	TQ_OP_STATS        = 9,
	TQ_OP_SHUTDOWN     = 10,
	TQ_OP_ACK          = 100,
	TQ_OP_ERROR        = 101,
	TQ_OP_RESULT       = 102
} tq_opcode;

#define TQ_SCHEMA_VERSION 1
#define TQ_MAX_FRAME (64u * 1024u * 1024u)

/* tq_conn is an opaque connection handle. */
typedef struct tq_conn tq_conn;

/* One search result row. */
typedef struct {
	uint64_t tid;
	float    score;
} tq_result;

/* Connection lifecycle. Returns NULL and sets *err (caller-owned static
 * string) on failure. */
tq_conn *tq_connect(const char *socket_path, char **err);
void     tq_close(tq_conn *c);

/* Returns the last error string for the connection (never NULL). */
const char *tq_last_error(tq_conn *c);

/* Operations. All return 0 on success, -1 on error (see tq_last_error). */
int tq_build_begin(tq_conn *c, const char *collection, uint32_t dim,
                   uint8_t bit_width, uint64_t rotator_seed);
int tq_build_vector(tq_conn *c, uint64_t tid, const float *values, uint32_t dim);
int tq_insert(tq_conn *c, uint64_t tid, const float *values, uint32_t dim);
int tq_delete(tq_conn *c, uint64_t tid);
int tq_build_commit(tq_conn *c);

/*
 * tq_search runs a query and returns results via the caller-allocated buffer
 * out[0..max_out). Writes the number of rows to *n_out. Returns 0 on success.
 */
int tq_search(tq_conn *c, const char *collection, const float *query, uint32_t dim,
              uint32_t top_k, float oversearch, int rerank, int exact,
              tq_result *out, uint32_t max_out, uint32_t *n_out);

/* Statistics reply. */
typedef struct {
	uint64_t vector_count;
	uint32_t sealed_segments;
	uint32_t growing_segment;
	uint64_t pinned_bytes;
} tq_stats;

int tq_stats_query(tq_conn *c, tq_stats *out);

#endif /* TURBODB_IPC_H */
