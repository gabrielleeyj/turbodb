/*
 * turbodb_ipc_probe.c — a small standalone harness that exercises the C IPC
 * client against a running engine (or the Go test server). Used by the
 * cross-language wire-compatibility test in internal/pgipc.
 *
 * Usage: turbodb_ipc_probe <socket_path>
 * Exits 0 on success, prints "OK tid=<n> count=<n> vcount=<n>" to stdout.
 */
#include "turbodb_ipc.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
	if (argc < 2) {
		fprintf(stderr, "usage: %s <socket_path>\n", argv[0]);
		return 2;
	}
	char *err = NULL;
	tq_conn *c = tq_connect(argv[1], &err);
	if (c == NULL) {
		fprintf(stderr, "connect: %s\n", err ? err : "unknown");
		return 1;
	}

	const uint32_t dim = 8;

	if (tq_build_begin(c, "probe", dim, 4, 0) != 0) {
		fprintf(stderr, "build_begin: %s\n", tq_last_error(c));
		return 1;
	}
	/* Insert distinct vectors so a self-query has a deterministic top match. */
	float query[8];
	for (uint64_t t = 1; t <= 10; t++) {
		float v[8];
		for (uint32_t i = 0; i < dim; i++)
			v[i] = (float) ((t * 13 + i) % 7);
		if (t == 1)
			for (uint32_t i = 0; i < dim; i++)
				query[i] = v[i];
		if (tq_build_vector(c, t, v, dim) != 0) {
			fprintf(stderr, "build_vector: %s\n", tq_last_error(c));
			return 1;
		}
	}
	if (tq_build_commit(c) != 0) {
		fprintf(stderr, "build_commit: %s\n", tq_last_error(c));
		return 1;
	}

	tq_result results[16];
	uint32_t n = 0;
	if (tq_search(c, "probe", query, dim, 5, 2.0f, 0, 0, results, 16, &n) != 0) {
		fprintf(stderr, "search: %s\n", tq_last_error(c));
		return 1;
	}

	tq_stats st;
	if (tq_stats_query(c, &st) != 0) {
		fprintf(stderr, "stats: %s\n", tq_last_error(c));
		return 1;
	}

	uint64_t top_tid = n > 0 ? results[0].tid : 0;
	printf("OK tid=%llu count=%u vcount=%llu\n",
	       (unsigned long long) top_tid, n, (unsigned long long) st.vector_count);

	tq_close(c);
	return 0;
}
