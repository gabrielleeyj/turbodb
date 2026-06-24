-- pg_turboquant 0.1 — GPU-accelerated TurboQuant vector index.
-- Registers a `turboquant` index access method that wraps pgvector's `vector`
-- type and delegates indexing/search to the turbodb-engine daemon.

-- Guard against running the script directly (must use CREATE EXTENSION).
\echo Use "CREATE EXTENSION pg_turboquant" to load this file. \quit

CREATE FUNCTION turboquant_handler(internal)
RETURNS index_am_handler
AS 'MODULE_PATHNAME'
LANGUAGE C;

CREATE ACCESS METHOD turboquant
TYPE INDEX
HANDLER turboquant_handler;

COMMENT ON ACCESS METHOD turboquant IS
'GPU-accelerated TurboQuant vector index (delegates to turbodb-engine)';

-- Operator class for L2 / inner-product ordering over pgvector's `vector`.
-- The access method only needs the ORDER BY distance operator (strategy 1);
-- it carries no support functions (amsupport = 0).
CREATE OPERATOR CLASS vector_turboquant_ops
DEFAULT FOR TYPE vector USING turboquant AS
	OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops;

-- Diagnostics view: per-index statistics sourced from the engine.
-- (Populated by a SQL function in a future revision; defined here so the
-- catalog object exists.)
CREATE VIEW pg_turboquant_indexes AS
	SELECT
		c.relname        AS index_name,
		c.oid            AS index_oid,
		'pgidx_' || c.oid::text AS engine_collection
	FROM pg_class c
	JOIN pg_am a ON a.oid = c.relam
	WHERE a.amname = 'turboquant';
