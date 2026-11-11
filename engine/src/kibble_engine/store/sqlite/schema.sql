-- Kibble index schema, version 1.
--
-- Applied once to a fresh database (see store.py). The version lives in
-- PRAGMA user_version, which is written in the same transaction as the DDL,
-- so a half-created schema is impossible. There are no migrations yet: while
-- pre-alpha, a version mismatch means "delete .kibble/ and re-index".
--
-- Not in this file: the vec0 table (chunk_vecs). Its dimension depends on the
-- configured embedder, so the store creates it at runtime.
--
-- Ownership rule: every row derived from a file points back to files(path)
-- with ON DELETE CASCADE. Replacing a file = delete its files row, re-insert.

BEGIN;

-- Key/value metadata: embedder_id, embed_dim, vec_index_valid, and anything
-- the core wants to remember (model name, base URL host, last commit...).
CREATE TABLE meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- Manifest: what the per-query stat sweep compares against.
CREATE TABLE files (
  path         TEXT PRIMARY KEY,            -- repo-relative, forward slashes
  content_hash TEXT NOT NULL,
  mtime_ns     INTEGER NOT NULL,
  size         INTEGER NOT NULL,
  language     TEXT,
  tags         TEXT NOT NULL DEFAULT '',    -- e.g. 'test generated vendor'
  indexed_at   INTEGER NOT NULL             -- unix seconds
);

CREATE TABLE chunks (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,  -- AUTOINCREMENT: ids never reused,
                                                    -- so stale FTS/vec rows can't alias
  file_path     TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
  chunk_hash    TEXT NOT NULL,              -- hash of embed_text
  kind          TEXT NOT NULL,              -- function, method, class, block...
  symbol        TEXT,
  parent_symbol TEXT,
  start_line    INTEGER NOT NULL,
  end_line      INTEGER NOT NULL,
  content       TEXT NOT NULL,              -- raw code, returned to the agent
  embed_text    TEXT NOT NULL,              -- enriched text that was embedded
  symbol_terms  TEXT NOT NULL DEFAULT ''    -- pre-split identifiers, for FTS
);
CREATE INDEX idx_chunks_file ON chunks(file_path);
CREATE INDEX idx_chunks_hash ON chunks(chunk_hash);

-- Canonical embeddings (unit-normalised float32 little-endian blobs).
-- chunk_vecs is only an index over these and can always be rebuilt.
-- Keyed by embedder_id so switching models never mixes vectors, and
-- switching back is free.
CREATE TABLE embedding_cache (
  chunk_hash  TEXT NOT NULL,
  embedder_id TEXT NOT NULL,
  embedding   BLOB NOT NULL,
  created_at  INTEGER NOT NULL,
  PRIMARY KEY (chunk_hash, embedder_id)
) WITHOUT ROWID;

-- Keyword search. External-content table: it indexes chunks without
-- duplicating the text, but the store must keep it in sync explicitly
-- (see replace_file). Column order matters for bm25() weights.
CREATE VIRTUAL TABLE chunks_fts USING fts5(
  symbol_terms,
  content,
  content='chunks',
  content_rowid='id'
);

-- Graph (empty during the vector-only PoC).
CREATE TABLE symbols (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  file_path  TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
  chunk_id   INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
  name       TEXT NOT NULL,
  kind       TEXT NOT NULL,
  start_line INTEGER NOT NULL,
  end_line   INTEGER NOT NULL
);
CREATE INDEX idx_symbols_name ON symbols(name);
CREATE INDEX idx_symbols_file ON symbols(file_path);

-- Edges are owned by their source file. The target is stored as an
-- unresolved name; target_symbol is filled by resolve_edges() when the name
-- maps to exactly one symbol, and set back to NULL if that symbol's file is
-- replaced.
CREATE TABLE edges (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  source_file   TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
  source_symbol INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
  kind          TEXT NOT NULL,              -- calls, imports, inherits
  target_name   TEXT NOT NULL,
  target_symbol INTEGER REFERENCES symbols(id) ON DELETE SET NULL
);
CREATE INDEX idx_edges_source_file ON edges(source_file);
CREATE INDEX idx_edges_source_symbol ON edges(source_symbol);
CREATE INDEX idx_edges_target_name ON edges(target_name);
CREATE INDEX idx_edges_target_symbol ON edges(target_symbol);

PRAGMA user_version = 1;

COMMIT;
