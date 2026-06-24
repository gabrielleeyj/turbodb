/*
 * pg_turboquant.c — PostgreSQL access method that delegates GPU-accelerated
 * TurboQuant vector indexing to the turbodb-engine daemon over a Unix-socket
 * IPC channel (see turbodb_ipc.h and internal/pgproto).
 *
 * The access method wraps pgvector's `vector` type: it indexes the same column
 * data but stores the quantized index in the engine, not in PostgreSQL heap
 * pages. Query-time scans send the ORDER BY query vector to the engine and
 * stream back heap TIDs ordered by similarity.
 *
 * Targets PostgreSQL 16/17/18 (developed against 18). Build with PGXS.
 */
#include "postgres.h"

#include "access/amapi.h"
#include "access/reloptions.h"
#include "access/relscan.h"
#include "access/tableam.h"
#include "catalog/index.h"
#include "commands/vacuum.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "nodes/pathnodes.h"
#include "storage/itemptr.h"
#include "utils/guc.h"
#include "utils/rel.h"
#include "utils/selfuncs.h"

#include "turbodb_ipc.h"

PG_MODULE_MAGIC;

/* GUCs */
static char *engine_socket = NULL;
#define DEFAULT_ENGINE_SOCKET "/var/run/turbodb/engine.sock"

/* Per-index reloptions. */
typedef struct TurboQuantOptions
{
	int32	vl_len_;		/* varlena header (do not touch directly) */
	int		bits;			/* quantizer bit width */
	double	oversearch_factor;
	bool	use_qjl;
} TurboQuantOptions;

static relopt_kind turboquant_relopt_kind;

/* ---- pgvector Vector layout (replicated to avoid a header dependency) ---- */
typedef struct PGVector
{
	int32	vl_len_;		/* varlena header */
	int16	dim;
	int16	unused;
	float4	x[FLEXIBLE_ARRAY_MEMBER];
} PGVector;

#define DatumGetPGVector(d) ((PGVector *) PG_DETOAST_DATUM(d))

/* Per-scan state. */
typedef struct TurboScanState
{
	tq_conn	   *conn;
	tq_result  *results;
	uint32		count;
	uint32		cursor;
} TurboScanState;

/* Compose a stable collection name for an index relation. */
static void
collection_name(Relation index, char *buf, size_t buflen)
{
	snprintf(buf, buflen, "pgidx_%u", RelationGetRelid(index));
}

/* Encode/decode a heap ItemPointer as a uint64 TID. */
static uint64
tid_encode(ItemPointer tid)
{
	return ((uint64) ItemPointerGetBlockNumber(tid) << 16) |
		   (uint64) ItemPointerGetOffsetNumber(tid);
}

static void
tid_decode(uint64 v, ItemPointer out)
{
	BlockNumber blk = (BlockNumber) (v >> 16);
	OffsetNumber off = (OffsetNumber) (v & 0xffff);
	ItemPointerSet(out, blk, off);
}

/* Connect to the engine or raise an error. */
static tq_conn *
connect_engine(void)
{
	char	   *err = NULL;
	tq_conn	   *c = tq_connect(engine_socket, &err);

	if (c == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_CONNECTION_FAILURE),
				 errmsg("pg_turboquant: cannot connect to engine at \"%s\": %s",
						engine_socket, err ? err : "unknown error")));
	return c;
}

/* Extract the bit width from an index's reloptions, defaulting to 4. */
static int
index_bits(Relation index)
{
	TurboQuantOptions *opts = (TurboQuantOptions *) index->rd_options;

	if (opts != NULL && opts->bits >= 1 && opts->bits <= 8)
		return opts->bits;
	return 4;
}

/* ---------------------------- build ---------------------------- */

typedef struct BuildState
{
	tq_conn	   *conn;
	char		collection[NAMEDATALEN + 16];
	int			dim;
	int			bits;
	bool		begun;
	double		tuples;
} BuildState;

static void
ensure_begun(BuildState *bs, int dim)
{
	if (bs->begun)
		return;
	bs->dim = dim;
	if (tq_build_begin(bs->conn, bs->collection, (uint32) dim, (uint8) bs->bits, 0) != 0)
		ereport(ERROR,
				(errmsg("pg_turboquant: build_begin failed: %s", tq_last_error(bs->conn))));
	bs->begun = true;
}

static void
build_callback(Relation index, ItemPointer tid, Datum *values,
			   bool *isnull, bool tupleIsAlive, void *state)
{
	BuildState *bs = (BuildState *) state;
	PGVector   *vec;

	if (!tupleIsAlive || isnull[0])
		return;

	vec = DatumGetPGVector(values[0]);
	ensure_begun(bs, vec->dim);
	if (vec->dim != bs->dim)
		ereport(ERROR,
				(errmsg("pg_turboquant: inconsistent vector dimension %d (expected %d)",
						vec->dim, bs->dim)));

	if (tq_build_vector(bs->conn, tid_encode(tid), vec->x, (uint32) vec->dim) != 0)
		ereport(ERROR,
				(errmsg("pg_turboquant: build_vector failed: %s", tq_last_error(bs->conn))));
	bs->tuples += 1;
}

static IndexBuildResult *
tq_ambuild(Relation heap, Relation index, struct IndexInfo *indexInfo)
{
	IndexBuildResult *result;
	BuildState	bs = {0};

	bs.conn = connect_engine();
	bs.bits = index_bits(index);
	collection_name(index, bs.collection, sizeof(bs.collection));

	PG_TRY();
	{
		(void) table_index_build_scan(heap, index, indexInfo, true, true,
									  build_callback, &bs, NULL);
		if (!bs.begun)
		{
			/* No live rows; create an empty collection with declared dim. */
			ensure_begun(&bs, TupleDescAttr(RelationGetDescr(index), 0)->atttypmod);
		}
		if (tq_build_commit(bs.conn) != 0)
			ereport(ERROR,
					(errmsg("pg_turboquant: build_commit failed: %s", tq_last_error(bs.conn))));
	}
	PG_FINALLY();
	{
		tq_close(bs.conn);
	}
	PG_END_TRY();

	result = (IndexBuildResult *) palloc0(sizeof(IndexBuildResult));
	result->heap_tuples = bs.tuples;
	result->index_tuples = bs.tuples;
	return result;
}

static void
tq_ambuildempty(Relation index)
{
	tq_conn	   *conn = connect_engine();
	char		collection[NAMEDATALEN + 16];
	int			dim = TupleDescAttr(RelationGetDescr(index), 0)->atttypmod;

	collection_name(index, collection, sizeof(collection));
	if (tq_build_begin(conn, collection, (uint32) (dim > 0 ? dim : 1),
					   (uint8) index_bits(index), 0) != 0 ||
		tq_build_commit(conn) != 0)
	{
		char errcopy[256];
		snprintf(errcopy, sizeof(errcopy), "%s", tq_last_error(conn));
		tq_close(conn);
		ereport(ERROR, (errmsg("pg_turboquant: buildempty failed: %s", errcopy)));
	}
	tq_close(conn);
}

static bool
tq_aminsert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid,
			Relation heap, IndexUniqueCheck checkUnique, bool indexUnchanged,
			struct IndexInfo *indexInfo)
{
	tq_conn	   *conn;
	PGVector   *vec;
	char		collection[NAMEDATALEN + 16];

	if (isnull[0])
		return false;

	conn = connect_engine();
	collection_name(index, collection, sizeof(collection));
	vec = DatumGetPGVector(values[0]);

	if (tq_insert(conn, tid_encode(heap_tid), vec->x, (uint32) vec->dim) != 0)
	{
		char errcopy[256];
		snprintf(errcopy, sizeof(errcopy), "%s", tq_last_error(conn));
		tq_close(conn);
		ereport(ERROR, (errmsg("pg_turboquant: insert failed: %s", errcopy)));
	}
	tq_close(conn);
	return true;
}

/* ---------------------------- vacuum ---------------------------- */

static IndexBulkDeleteResult *
tq_ambulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats,
				IndexBulkDeleteCallback callback, void *callback_state)
{
	if (stats == NULL)
		stats = (IndexBulkDeleteResult *) palloc0(sizeof(IndexBulkDeleteResult));
	/*
	 * The engine owns segment storage; deletions are tombstoned via aminsert's
	 * counterpart at the SQL/trigger level. A full implementation would scan
	 * engine TIDs and call the VACUUM callback; deferred until the engine
	 * exposes a TID enumeration RPC.
	 */
	return stats;
}

static IndexBulkDeleteResult *
tq_amvacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats)
{
	return stats;
}

/* ---------------------------- scan ---------------------------- */

#define MAX_SCAN_RESULTS 1024

static IndexScanDesc
tq_ambeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan = RelationGetIndexScan(index, nkeys, norderbys);
	TurboScanState *st = (TurboScanState *) palloc0(sizeof(TurboScanState));

	st->results = (tq_result *) palloc0(sizeof(tq_result) * MAX_SCAN_RESULTS);
	scan->opaque = st;
	return scan;
}

static void
tq_amrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
	TurboScanState *st = (TurboScanState *) scan->opaque;
	PGVector   *query;
	char		collection[NAMEDATALEN + 16];
	uint32		n = 0;

	st->cursor = 0;
	st->count = 0;

	if (norderbys < 1)
		ereport(ERROR,
				(errmsg("pg_turboquant: scan requires an ORDER BY distance clause")));

	query = DatumGetPGVector(orderbys[0].sk_argument);
	collection_name(scan->indexRelation, collection, sizeof(collection));

	if (st->conn == NULL)
		st->conn = connect_engine();

	if (tq_search(st->conn, collection, query->x, (uint32) query->dim,
				  MAX_SCAN_RESULTS, 2.0f, 0, 0, st->results, MAX_SCAN_RESULTS, &n) != 0)
		ereport(ERROR,
				(errmsg("pg_turboquant: search failed: %s", tq_last_error(st->conn))));
	st->count = n;
}

static bool
tq_amgettuple(IndexScanDesc scan, ScanDirection direction)
{
	TurboScanState *st = (TurboScanState *) scan->opaque;

	if (st->cursor >= st->count)
		return false;

	tid_decode(st->results[st->cursor].tid, &scan->xs_heaptid);
	st->cursor++;
	scan->xs_recheckorderby = false;
	return true;
}

static void
tq_amendscan(IndexScanDesc scan)
{
	TurboScanState *st = (TurboScanState *) scan->opaque;

	if (st != NULL && st->conn != NULL)
		tq_close(st->conn);
}

/* ---------------------------- planning ---------------------------- */

static void
tq_amcostestimate(struct PlannerInfo *root, struct IndexPath *path, double loop_count,
				  Cost *indexStartupCost, Cost *indexTotalCost,
				  Selectivity *indexSelectivity, double *indexCorrelation,
				  double *indexPages)
{
	GenericCosts costs;

	memset(&costs, 0, sizeof(costs));
	genericcostestimate(root, path, loop_count, &costs);

	/* GPU dispatch is fast; bias the planner toward this index for ORDER BY. */
	*indexStartupCost = costs.indexStartupCost;
	*indexTotalCost = costs.indexTotalCost * 0.25;
	*indexSelectivity = costs.indexSelectivity;
	*indexCorrelation = costs.indexCorrelation;
	*indexPages = costs.numIndexPages;
}

static bytea *
tq_amoptions(Datum reloptions, bool validate)
{
	static const relopt_parse_elt tab[] = {
		{"bits", RELOPT_TYPE_INT, offsetof(TurboQuantOptions, bits)},
		{"oversearch_factor", RELOPT_TYPE_REAL, offsetof(TurboQuantOptions, oversearch_factor)},
		{"use_qjl", RELOPT_TYPE_BOOL, offsetof(TurboQuantOptions, use_qjl)},
	};

	return (bytea *) build_reloptions(reloptions, validate, turboquant_relopt_kind,
									  sizeof(TurboQuantOptions), tab, lengthof(tab));
}

static bool
tq_amvalidate(Oid opclassoid)
{
	return true;
}

/* ---------------------------- handler ---------------------------- */

PG_FUNCTION_INFO_V1(turboquant_handler);

Datum
turboquant_handler(PG_FUNCTION_ARGS)
{
	IndexAmRoutine *amroutine = makeNode(IndexAmRoutine);

	amroutine->amstrategies = 0;
	amroutine->amsupport = 0;
	amroutine->amoptsprocnum = 0;
	amroutine->amcanorder = false;
	amroutine->amcanorderbyop = true;	/* supports ORDER BY distance op */
	amroutine->amcanhash = false;
	amroutine->amconsistentequality = false;
	amroutine->amconsistentordering = false;
	amroutine->amcanbackward = false;
	amroutine->amcanunique = false;
	amroutine->amcanmulticol = false;
	amroutine->amoptionalkey = true;
	amroutine->amsearcharray = false;
	amroutine->amsearchnulls = false;
	amroutine->amstorage = false;
	amroutine->amclusterable = false;
	amroutine->ampredlocks = false;
	amroutine->amcanparallel = false;
	amroutine->amcanbuildparallel = false;
	amroutine->amcaninclude = false;
	amroutine->amusemaintenanceworkmem = false;
	amroutine->amsummarizing = false;
	amroutine->amparallelvacuumoptions = VACUUM_OPTION_NO_PARALLEL;
	amroutine->amkeytype = InvalidOid;

	amroutine->ambuild = tq_ambuild;
	amroutine->ambuildempty = tq_ambuildempty;
	amroutine->aminsert = tq_aminsert;
	amroutine->aminsertcleanup = NULL;
	amroutine->ambulkdelete = tq_ambulkdelete;
	amroutine->amvacuumcleanup = tq_amvacuumcleanup;
	amroutine->amcanreturn = NULL;
	amroutine->amcostestimate = tq_amcostestimate;
	amroutine->amgettreeheight = NULL;
	amroutine->amoptions = tq_amoptions;
	amroutine->amproperty = NULL;
	amroutine->ambuildphasename = NULL;
	amroutine->amvalidate = tq_amvalidate;
	amroutine->amadjustmembers = NULL;
	amroutine->ambeginscan = tq_ambeginscan;
	amroutine->amrescan = tq_amrescan;
	amroutine->amgettuple = tq_amgettuple;
	amroutine->amgetbitmap = NULL;
	amroutine->amendscan = tq_amendscan;
	amroutine->ammarkpos = NULL;
	amroutine->amrestrpos = NULL;
	amroutine->amestimateparallelscan = NULL;
	amroutine->aminitparallelscan = NULL;
	amroutine->amparallelrescan = NULL;
	amroutine->amtranslatestrategy = NULL;
	amroutine->amtranslatecmptype = NULL;

	PG_RETURN_POINTER(amroutine);
}

void
_PG_init(void)
{
	DefineCustomStringVariable("pg_turboquant.engine_socket",
							   "Unix socket path of the turbodb engine IPC server.",
							   NULL,
							   &engine_socket,
							   DEFAULT_ENGINE_SOCKET,
							   PGC_SIGHUP,
							   0,
							   NULL, NULL, NULL);

	turboquant_relopt_kind = add_reloption_kind();
	add_int_reloption(turboquant_relopt_kind, "bits",
					  "Quantizer bit width (1-8).", 4, 1, 8, AccessExclusiveLock);
	add_real_reloption(turboquant_relopt_kind, "oversearch_factor",
					   "Per-segment oversearch multiplier.", 2.0, 1.0, 64.0, AccessExclusiveLock);
	add_bool_reloption(turboquant_relopt_kind, "use_qjl",
					   "Use the QJL inner-product sketch.", true, AccessExclusiveLock);

	MarkGUCPrefixReserved("pg_turboquant");
}
