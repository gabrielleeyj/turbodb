/*
 * turbodb_ipc.c — implementation of the TurboDB engine IPC client.
 *
 * Self-contained (libc + POSIX). See turbodb_ipc.h for the protocol summary
 * and internal/pgproto for the Go counterpart.
 */
#include "turbodb_ipc.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#define TQ_ERRBUF 256

struct tq_conn {
	int  fd;
	char err[TQ_ERRBUF];
};

static void set_err(tq_conn *c, const char *msg)
{
	if (c == NULL)
		return;
	snprintf(c->err, sizeof(c->err), "%s", msg);
}

static void set_errno(tq_conn *c, const char *prefix)
{
	if (c == NULL)
		return;
	snprintf(c->err, sizeof(c->err), "%s: %s", prefix, strerror(errno));
}

const char *tq_last_error(tq_conn *c)
{
	if (c == NULL)
		return "null connection";
	return c->err;
}

tq_conn *tq_connect(const char *socket_path, char **err)
{
	static char static_err[TQ_ERRBUF];
	struct sockaddr_un addr;
	tq_conn *c;

	if (socket_path == NULL || socket_path[0] == '\0') {
		if (err) { snprintf(static_err, sizeof(static_err), "empty socket path"); *err = static_err; }
		return NULL;
	}
	if (strlen(socket_path) >= sizeof(addr.sun_path)) {
		if (err) { snprintf(static_err, sizeof(static_err), "socket path too long"); *err = static_err; }
		return NULL;
	}

	c = (tq_conn *) calloc(1, sizeof(*c));
	if (c == NULL) {
		if (err) { snprintf(static_err, sizeof(static_err), "out of memory"); *err = static_err; }
		return NULL;
	}
	c->fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (c->fd < 0) {
		set_errno(c, "socket");
		if (err) *err = c->err;
		free(c);
		return NULL;
	}

	memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_UNIX;
	strncpy(addr.sun_path, socket_path, sizeof(addr.sun_path) - 1);
	if (connect(c->fd, (struct sockaddr *) &addr, sizeof(addr)) < 0) {
		set_errno(c, "connect");
		if (err) *err = c->err;
		close(c->fd);
		free(c);
		return NULL;
	}
	return c;
}

void tq_close(tq_conn *c)
{
	if (c == NULL)
		return;
	if (c->fd >= 0)
		close(c->fd);
	free(c);
}

/* Reliable write/read helpers (handle partial transfers). */
static int write_all(tq_conn *c, const uint8_t *buf, size_t len)
{
	size_t off = 0;
	while (off < len) {
		ssize_t n = write(c->fd, buf + off, len - off);
		if (n < 0) {
			if (errno == EINTR)
				continue;
			set_errno(c, "write");
			return -1;
		}
		off += (size_t) n;
	}
	return 0;
}

static int read_all(tq_conn *c, uint8_t *buf, size_t len)
{
	size_t off = 0;
	while (off < len) {
		ssize_t n = read(c->fd, buf + off, len - off);
		if (n == 0) {
			set_err(c, "connection closed by engine");
			return -1;
		}
		if (n < 0) {
			if (errno == EINTR)
				continue;
			set_errno(c, "read");
			return -1;
		}
		off += (size_t) n;
	}
	return 0;
}

/* Little-endian append helpers operating on a growable buffer. */
typedef struct {
	uint8_t *data;
	size_t   len;
	size_t   cap;
	int      oom;
} tq_buf;

static void buf_reserve(tq_buf *b, size_t extra)
{
	size_t cap;
	uint8_t *p;

	if (b->oom)
		return;
	if (b->len + extra <= b->cap)
		return;
	cap = b->cap ? b->cap * 2 : 64;
	while (cap < b->len + extra)
		cap *= 2;
	p = (uint8_t *) realloc(b->data, cap);
	if (p == NULL) { b->oom = 1; return; }
	b->data = p;
	b->cap = cap;
}

static void buf_u8(tq_buf *b, uint8_t v)  { buf_reserve(b, 1); if (!b->oom) b->data[b->len++] = v; }
static void buf_u16le(tq_buf *b, uint16_t v) { buf_u8(b, v & 0xff); buf_u8(b, (v >> 8) & 0xff); }
static void buf_u32le(tq_buf *b, uint32_t v)
{
	buf_u8(b, v & 0xff); buf_u8(b, (v >> 8) & 0xff);
	buf_u8(b, (v >> 16) & 0xff); buf_u8(b, (v >> 24) & 0xff);
}
static void buf_u64le(tq_buf *b, uint64_t v)
{
	for (int i = 0; i < 8; i++)
		buf_u8(b, (uint8_t) ((v >> (8 * i)) & 0xff));
}
static void buf_f32le(tq_buf *b, float f)
{
	uint32_t bits;
	memcpy(&bits, &f, sizeof(bits));
	buf_u32le(b, bits);
}
static void buf_str(tq_buf *b, const char *s)
{
	uint32_t n = (uint32_t) strlen(s);
	buf_u32le(b, n);
	buf_reserve(b, n);
	if (!b->oom) { memcpy(b->data + b->len, s, n); b->len += n; }
}
static void buf_vec(tq_buf *b, const float *v, uint32_t dim)
{
	buf_u32le(b, dim);
	for (uint32_t i = 0; i < dim; i++)
		buf_f32le(b, v[i]);
}

/* send_frame writes opcode + schema version + payload as one length-prefixed
 * frame. Frees payload buffer. */
static int send_frame(tq_conn *c, tq_opcode op, tq_buf *payload)
{
	uint32_t body;
	tq_buf frame = {0};
	int rc;

	if (payload->oom) {
		set_err(c, "out of memory building payload");
		free(payload->data);
		return -1;
	}
	body = (uint32_t) (2 + 2 + payload->len);
	if (body > TQ_MAX_FRAME) {
		set_err(c, "frame too large");
		free(payload->data);
		return -1;
	}
	buf_u8(&frame, (body >> 24) & 0xff);   /* big-endian length */
	buf_u8(&frame, (body >> 16) & 0xff);
	buf_u8(&frame, (body >> 8) & 0xff);
	buf_u8(&frame, body & 0xff);
	buf_u8(&frame, (op >> 8) & 0xff);      /* big-endian opcode */
	buf_u8(&frame, op & 0xff);
	buf_u16le(&frame, TQ_SCHEMA_VERSION);  /* little-endian version */
	buf_reserve(&frame, payload->len);
	if (!frame.oom && payload->len) {
		memcpy(frame.data + frame.len, payload->data, payload->len);
		frame.len += payload->len;
	}
	free(payload->data);
	if (frame.oom) {
		set_err(c, "out of memory building frame");
		free(frame.data);
		return -1;
	}
	rc = write_all(c, frame.data, frame.len);
	free(frame.data);
	return rc;
}

/* recv_frame reads one frame, returning the opcode and a malloc'd payload body
 * (without the version prefix). Caller frees *payload. */
static int recv_frame(tq_conn *c, tq_opcode *op, uint8_t **payload, uint32_t *payload_len)
{
	uint8_t lenbuf[4];
	uint32_t body;
	uint8_t *buf;
	uint16_t version;

	if (read_all(c, lenbuf, 4) < 0)
		return -1;
	body = ((uint32_t) lenbuf[0] << 24) | ((uint32_t) lenbuf[1] << 16) |
	       ((uint32_t) lenbuf[2] << 8) | (uint32_t) lenbuf[3];
	if (body < 4 || body > TQ_MAX_FRAME) {
		set_err(c, "invalid frame length");
		return -1;
	}
	buf = (uint8_t *) malloc(body);
	if (buf == NULL) { set_err(c, "out of memory"); return -1; }
	if (read_all(c, buf, body) < 0) { free(buf); return -1; }

	*op = (tq_opcode) (((uint16_t) buf[0] << 8) | buf[1]);
	version = (uint16_t) buf[2] | ((uint16_t) buf[3] << 8);
	if (version != TQ_SCHEMA_VERSION) {
		set_err(c, "unsupported schema version");
		free(buf);
		return -1;
	}
	*payload_len = body - 4;
	*payload = (uint8_t *) malloc(*payload_len ? *payload_len : 1);
	if (*payload == NULL) { free(buf); set_err(c, "out of memory"); return -1; }
	memcpy(*payload, buf + 4, *payload_len);
	free(buf);
	return 0;
}

/* expect_reply reads a frame and maps OP_ERROR to -1 with the message. */
static int expect_reply(tq_conn *c, tq_opcode want, uint8_t **payload, uint32_t *payload_len)
{
	tq_opcode op;
	uint8_t *p = NULL;
	uint32_t plen = 0;
	if (recv_frame(c, &op, &p, &plen) < 0)
		return -1;
	if (op == TQ_OP_ERROR) {
		/* payload = u32 len + string */
		if (plen >= 4) {
			uint32_t n = (uint32_t) p[0] | ((uint32_t) p[1] << 8) |
			             ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
			if (4 + n <= plen) {
				char msg[TQ_ERRBUF];
				uint32_t copy = n < sizeof(msg) - 1 ? n : (uint32_t) sizeof(msg) - 1;
				memcpy(msg, p + 4, copy);
				msg[copy] = '\0';
				set_err(c, msg);
			}
		}
		free(p);
		return -1;
	}
	if (op != want) {
		set_err(c, "unexpected reply opcode");
		free(p);
		return -1;
	}
	if (payload) { *payload = p; *payload_len = plen; }
	else free(p);
	return 0;
}

static int expect_ack(tq_conn *c)
{
	return expect_reply(c, TQ_OP_ACK, NULL, NULL);
}

int tq_build_begin(tq_conn *c, const char *collection, uint32_t dim,
                   uint8_t bit_width, uint64_t rotator_seed)
{
	tq_buf p = {0};
	buf_str(&p, collection);
	buf_u32le(&p, dim);
	buf_u8(&p, bit_width);
	buf_u8(&p, 0); /* variant MSE */
	buf_u64le(&p, rotator_seed);
	if (send_frame(c, TQ_OP_BUILD_BEGIN, &p) < 0)
		return -1;
	return expect_ack(c);
}

static int send_vector(tq_conn *c, tq_opcode op, uint64_t tid, const float *values, uint32_t dim)
{
	tq_buf p = {0};
	buf_u64le(&p, tid);
	buf_vec(&p, values, dim);
	if (send_frame(c, op, &p) < 0)
		return -1;
	return expect_ack(c);
}

int tq_build_vector(tq_conn *c, uint64_t tid, const float *values, uint32_t dim)
{
	return send_vector(c, TQ_OP_BUILD_VECTOR, tid, values, dim);
}

int tq_insert(tq_conn *c, uint64_t tid, const float *values, uint32_t dim)
{
	return send_vector(c, TQ_OP_INSERT, tid, values, dim);
}

int tq_delete(tq_conn *c, uint64_t tid)
{
	tq_buf p = {0};
	buf_u64le(&p, tid);
	if (send_frame(c, TQ_OP_DELETE, &p) < 0)
		return -1;
	return expect_ack(c);
}

int tq_build_commit(tq_conn *c)
{
	tq_buf p = {0};
	if (send_frame(c, TQ_OP_BUILD_COMMIT, &p) < 0)
		return -1;
	return expect_ack(c);
}

/* read helpers for fixed-width LE fields from a payload buffer */
static uint64_t rd_u64le(const uint8_t *p) {
	uint64_t v = 0;
	for (int i = 0; i < 8; i++) v |= (uint64_t) p[i] << (8 * i);
	return v;
}
static uint32_t rd_u32le(const uint8_t *p) {
	return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}
static float rd_f32le(const uint8_t *p) {
	uint32_t bits = rd_u32le(p);
	float f;
	memcpy(&f, &bits, sizeof(f));
	return f;
}

int tq_search(tq_conn *c, const char *collection, const float *query, uint32_t dim,
              uint32_t top_k, float oversearch, int rerank, int exact,
              tq_result *out, uint32_t max_out, uint32_t *n_out)
{
	tq_buf p = {0};
	uint32_t count = 0;
	tq_buf ep = {0};

	buf_str(&p, collection);
	buf_vec(&p, query, dim);
	buf_u32le(&p, top_k);
	buf_f32le(&p, oversearch);
	buf_u8(&p, rerank ? 1 : 0);
	buf_u8(&p, exact ? 1 : 0);
	if (send_frame(c, TQ_OP_SEARCH_BEGIN, &p) < 0)
		return -1;
	if (expect_ack(c) < 0)
		return -1;

	for (;;) {
		tq_buf np = {0};
		uint8_t *rp = NULL;
		uint32_t rlen = 0;
		uint64_t tid;
		float score;
		uint8_t done;

		if (send_frame(c, TQ_OP_SEARCH_NEXT, &np) < 0)
			return -1;
		if (expect_reply(c, TQ_OP_RESULT, &rp, &rlen) < 0)
			return -1;
		/* ResultMsg: u64 tid + f32 score + u8 done */
		if (rlen < 13) { free(rp); set_err(c, "short result"); return -1; }
		tid = rd_u64le(rp);
		score = rd_f32le(rp + 8);
		done = rp[12];
		free(rp);
		if (done)
			break;
		if (count < max_out) {
			out[count].tid = tid;
			out[count].score = score;
		}
		count++;
	}

	if (send_frame(c, TQ_OP_SEARCH_END, &ep) < 0)
		return -1;
	if (expect_ack(c) < 0)
		return -1;

	if (n_out)
		*n_out = count < max_out ? count : max_out;
	return 0;
}

int tq_stats_query(tq_conn *c, tq_stats *out)
{
	tq_buf p = {0};
	uint8_t *rp = NULL;
	uint32_t rlen = 0;

	if (send_frame(c, TQ_OP_STATS, &p) < 0)
		return -1;
	if (expect_reply(c, TQ_OP_STATS, &rp, &rlen) < 0)
		return -1;
	if (rlen < 24) { free(rp); set_err(c, "short stats reply"); return -1; }
	out->vector_count = rd_u64le(rp);
	out->sealed_segments = rd_u32le(rp + 8);
	out->growing_segment = rd_u32le(rp + 12);
	out->pinned_bytes = rd_u64le(rp + 16);
	free(rp);
	return 0;
}
