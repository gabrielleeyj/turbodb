-- Example source schema for turbodb-sync. Runs once on first container start.
-- Mirrors the schema the soak harness validates against; adjust the embedding
-- dimension to match your collection.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS public.docs (
    doc_id     text PRIMARY KEY,
    embedding  vector(8),
    deleted_at text
);

CREATE PUBLICATION turbodb_pub FOR TABLE public.docs;
