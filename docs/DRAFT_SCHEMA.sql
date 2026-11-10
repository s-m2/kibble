-- Per-connection pragmas (foreign_keys is NOT persistent, set it on every connection)
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- Metadata: schema_version, embed_model, embed_dim, embed_source,
-- chunker_version, last_indexed_commit
CREATE TABLE meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- Manifest: what the stat sweep compares against
CREATE TABLE files (
  path         TEXT PRIMARY KEY,        -- repo-relative, forward slashes
  content_hash TEXT NOT NULL,
  mtime_ns     INTEGER NOT NULL,
  size         INTEGER NOT NULL,
  language     TEXT,
  tags         TEXT,                    -- e.g. 'test', 'generated', 'vendor'
  indexed_at   INTEGER NOT NULL
);

CREATE TABLE chunks (
  id            INTEGER PRIMARY KEY,
  file_path     TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
  chunk_hash    TEXT NOT NULL,          -- hash of embed_text
  kind          TEXT NOT NULL,          -- function, method, class, block...
  symbol        TEXT,
  parent_symbol TEXT,
  start_line    INTEGER NOT NULL,
  end_line      INTEGER NOT NULL,
  content       TEXT NOT NULL,          -- raw code, returned to the agent
  embed_text    TEXT NOT NULL,          -- enriched text that gets embedded
  symbol_terms  TEXT                    -- pre-split identifiers for FTS
);
CREATE INDEX idx_chunks_file ON chunks(file_path);
CREATE INDEX idx_chunks_hash ON chunks(chunk_hash);

-- Canonical embeddings; the vec table below is a rebuildable index over these
CREATE TABLE embedding_cache (
  chunk_hash TEXT NOT NULL,
  model      TEXT NOT NULL,
  embedding  BLOB NOT NULL,
  last_used  INTEGER,
  PRIMARY KEY (chunk_hash, model)
);

-- Keyword search (external content: points at chunks, no duplicate text)
CREATE VIRTUAL TABLE chunks_fts USING fts5(
  symbol_terms, content,
  content='chunks', content_rowid='id'
);

-- Vector search; dimension is templated from meta.embed_dim at creation
CREATE VIRTUAL TABLE chunk_vecs USING vec0(embedding float[768]);

-- Graph tables (empty in the vector-only PoC)
CREATE TABLE symbols (
  id         INTEGER PRIMARY KEY,
  file_path  TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
  chunk_id   INTEGER,
  name       TEXT NOT NULL,
  kind       TEXT NOT NULL,
  start_line INTEGER,
  end_line   INTEGER
);
CREATE INDEX idx_symbols_name ON symbols(name);
CREATE INDEX idx_symbols_file ON symbols(file_path);

CREATE TABLE edges (
  id            INTEGER PRIMARY KEY,
  source_file   TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
  source_symbol INTEGER NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
  kind          TEXT NOT NULL,          -- calls, imports, inherits
  target_name   TEXT NOT NULL,          -- unresolved name
  target_symbol INTEGER REFERENCES symbols(id) ON DELETE SET NULL
);
CREATE INDEX idx_edges_source_file ON edges(source_file);
CREATE INDEX idx_edges_target_name ON edges(target_name);
CREATE INDEX idx_edges_target      ON edges(target_symbol);

