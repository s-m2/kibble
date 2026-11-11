"""SQLite implementation of the Store protocol.

Design notes (see docs/decisions/0002 and 0008):

- One connection, one lock. All public methods take the lock, so the store is
  safe to call from FastAPI's worker threads. Writes use BEGIN IMMEDIATE.
- `replace_file` is a single transaction over manifest, chunks, FTS, vectors,
  symbols and edges. Embeddings are computed by the caller beforehand; nothing
  slow happens while the write lock is held.
- `embedding_cache` is the canonical home of vectors. `chunk_vecs` (sqlite-vec)
  is an accelerator that can be rebuilt from it. If the extension cannot be
  loaded, a pure-Python scan over the cache gives identical rankings, only slower.
- Vectors are normalised to unit length on the way in, so dot product, cosine and
  L2 rankings all agree.
"""

from __future__ import annotations

import heapq
import logging
import math
import re
import sqlite3
import struct
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from importlib import resources
from operator import mul
from pathlib import Path
from typing import Literal

from kibble_engine.store.models import (
    CalleeRef,
    ChunkRecord,
    EdgeRecord,
    FileRecord,
    FileState,
    Hit,
    StoredChunk,
    SymbolRecord,
    SymbolRef,
)

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
VectorBackend = Literal["auto", "vec0", "python"]

_SQL_BATCH = 500  # stay well under SQLite's bound-variable limit


class StoreError(Exception):
    """Base class for store problems that need a human-readable message."""


class SchemaVersionError(StoreError):
    pass


class EmbedderMismatchError(StoreError):
    pass


class VectorBackendError(StoreError):
    pass


def _pack(vector: Sequence[float], dim: int) -> bytes:
    """Validate, normalise to unit length, and pack as float32 little-endian."""
    if len(vector) != dim:
        raise ValueError(f"embedding has {len(vector)} dimensions, store expects {dim}")
    norm = math.sqrt(sum(x * x for x in vector))
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("embedding must be finite and non-zero")
    return struct.pack(f"<{dim}f", *(x / norm for x in vector))


def _unpack(blob: bytes, dim: int) -> tuple[float, ...]:
    return struct.unpack(f"<{dim}f", blob)


def _fts_query(query: str, match_all: bool) -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    Agents send things like `auth.validate` or `user-id`, which are syntax errors
    in raw FTS5. Quote every term (doubling embedded quotes) and join with OR (or
    AND). Terms with no letters or digits are dropped.
    """
    terms = [t for t in query.split() if re.search(r"\w", t)]
    quoted = ['"' + t.replace('"', '""') + '"' for t in terms]
    return (" AND " if match_all else " OR ").join(quoted)


def _try_load_vec0(conn: sqlite3.Connection) -> bool:
    try:
        import sqlite_vec
    except ImportError:
        return False
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
    except (AttributeError, sqlite3.Error):
        return False
    finally:
        try:
            conn.enable_load_extension(False)
        except (AttributeError, sqlite3.Error):
            pass
    return True


def _check_fts5(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("CREATE VIRTUAL TABLE temp.kibble_fts_probe USING fts5(x)")
        conn.execute("DROP TABLE temp.kibble_fts_probe")
    except sqlite3.OperationalError as exc:
        raise StoreError(
            "This Python's SQLite was built without FTS5, which Kibble requires "
            f"(sqlite {sqlite3.sqlite_version}: {exc})."
        ) from exc


def _rowid(cursor: sqlite3.Cursor) -> int:
    if cursor.lastrowid is None:
        raise StoreError("insert did not produce a row id")
    return cursor.lastrowid


class SqliteStore:
    """Use `SqliteStore.open(...)`; the constructor is internal."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        embedder_id: str,
        embed_dim: int,
        use_vec0: bool,
    ) -> None:
        self._conn = conn
        self._lock = threading.RLock()
        self._embedder_id = embedder_id
        self._dim = embed_dim
        self._use_vec0 = use_vec0

    # ------------------------------------------------------------------ open

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        embedder_id: str,
        embed_dim: int,
        vector_backend: VectorBackend = "auto",
    ) -> SqliteStore:
        """Open (creating if needed) the index at `path` (or ':memory:')."""
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        path_str = str(path)
        if path_str != ":memory:":
            Path(path_str).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(path_str, isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA foreign_keys = ON")  # per-connection, not persistent
            conn.execute("PRAGMA busy_timeout = 5000")
            _check_fts5(conn)

            use_vec0 = False
            if vector_backend != "python":
                use_vec0 = _try_load_vec0(conn)
                if vector_backend == "vec0" and not use_vec0:
                    raise VectorBackendError(
                        "vector_backend='vec0' requested but the sqlite-vec extension "
                        "could not be loaded (is sqlite-vec installed, and does this "
                        "Python allow load_extension?)"
                    )
                if not use_vec0:
                    log.warning(
                        "sqlite-vec unavailable; falling back to pure-Python vector scan "
                        "(correct but slower on large indexes)"
                    )

            store = cls(conn, embedder_id=embedder_id, embed_dim=embed_dim, use_vec0=use_vec0)
            store._initialise()
            return store
        except BaseException:
            conn.close()
            raise

    def _initialise(self) -> None:
        version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
        if version == 0:
            self._conn.executescript(_schema_sql())
            self.set_meta("embedder_id", self._embedder_id)
            self.set_meta("embed_dim", str(self._dim))
        elif version != SCHEMA_VERSION:
            raise SchemaVersionError(
                f"index schema version is {version}, this build expects {SCHEMA_VERSION}. "
                "There are no migrations yet: delete the .kibble/ directory and re-index."
            )
        else:
            stored_id = self.get_meta("embedder_id")
            stored_dim = self.get_meta("embed_dim")
            if stored_id != self._embedder_id or stored_dim != str(self._dim):
                raise EmbedderMismatchError(
                    f"index was built with embedder {stored_id!r} (dim {stored_dim}); "
                    f"configured embedder is {self._embedder_id!r} (dim {self._dim}). "
                    "Call reset_embedder() (or re-index) to switch."
                )
        self._sync_vec_index()

    def _sync_vec_index(self) -> None:
        """Make sure chunk_vecs exists and is valid, or mark it stale in python mode."""
        if not self._use_vec0:
            # Python-mode writes don't maintain chunk_vecs; flag it so a later
            # vec0-mode open rebuilds it from the embedding cache.
            self.set_meta("vec_index_valid", "0")
            return
        exists = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'chunk_vecs'"
        ).fetchone()
        if not exists or self.get_meta("vec_index_valid") != "1":
            self._rebuild_vec_index()

    def _rebuild_vec_index(self) -> None:
        with self._transaction() as conn:
            conn.execute("DROP TABLE IF EXISTS chunk_vecs")
            conn.execute(
                f"CREATE VIRTUAL TABLE chunk_vecs USING vec0(embedding float[{self._dim}])"
            )
            rows = conn.execute(
                "SELECT c.id AS id, ec.embedding AS embedding FROM chunks c "
                "JOIN embedding_cache ec ON ec.chunk_hash = c.chunk_hash AND ec.embedder_id = ?",
                (self._embedder_id,),
            ).fetchall()
            conn.executemany(
                "INSERT INTO chunk_vecs(rowid, embedding) VALUES (?, ?)",
                [(r["id"], r["embedding"]) for r in rows],
            )
            conn.execute(
                "INSERT INTO meta(key, value) VALUES ('vec_index_valid', '1') "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
            )

    # --------------------------------------------------------------- plumbing

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> SqliteStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def vector_backend(self) -> str:
        return "vec0" if self._use_vec0 else "python"

    # ------------------------------------------------------------------- meta

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {
                table: int(self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in ("files", "chunks", "symbols", "edges", "embedding_cache")
            }

    # ------------------------------------------------------------ write path

    def replace_file(
        self,
        file: FileRecord,
        chunks: Sequence[ChunkRecord],
        symbols: Sequence[SymbolRecord] = (),
        edges: Sequence[EdgeRecord] = (),
    ) -> None:
        # Validate and pack everything *before* touching the database, so bad
        # input can never leave a half-written file behind.
        self._validate(file, chunks, symbols, edges)
        blobs = [_pack(c.embedding, self._dim) for c in chunks]

        with self._transaction() as conn:
            self._purge_file(conn, file.path)
            chunk_ids = self._insert_file(conn, file, chunks, blobs)
            self._insert_graph(conn, file.path, chunk_ids, symbols, edges)

    def remove_file(self, path: str) -> None:
        with self._transaction() as conn:
            self._purge_file(conn, path)

    @staticmethod
    def _validate(
        file: FileRecord,
        chunks: Sequence[ChunkRecord],
        symbols: Sequence[SymbolRecord],
        edges: Sequence[EdgeRecord],
    ) -> None:
        if not file.path or file.path.startswith("/") or "\\" in file.path:
            raise ValueError(f"path must be repo-relative with forward slashes: {file.path!r}")
        for c in chunks:
            if c.start_line > c.end_line:
                raise ValueError(f"chunk has start_line > end_line: {c.start_line}>{c.end_line}")
        for s in symbols:
            if s.chunk_index is not None and not 0 <= s.chunk_index < len(chunks):
                raise ValueError(f"symbol {s.name!r} has chunk_index out of range")
        for e in edges:
            if not 0 <= e.source_index < len(symbols):
                raise ValueError(f"edge to {e.target_name!r} has source_index out of range")

    def _purge_file(self, conn: sqlite3.Connection, path: str) -> None:
        """Delete everything derived from `path`, in an order that keeps FTS/vec in sync.

        FTS5 external-content tables must be told the *old* values to remove, so
        they are read from `chunks` before the cascade deletes those rows. Virtual
        tables don't take part in foreign-key cascades, so vectors go explicitly.
        """
        rows = conn.execute(
            "SELECT id, symbol_terms, content FROM chunks WHERE file_path = ?", (path,)
        ).fetchall()
        if rows:
            conn.executemany(
                "INSERT INTO chunks_fts(chunks_fts, rowid, symbol_terms, content) "
                "VALUES ('delete', ?, ?, ?)",
                [(r["id"], r["symbol_terms"], r["content"]) for r in rows],
            )
            if self._use_vec0:
                conn.executemany(
                    "DELETE FROM chunk_vecs WHERE rowid = ?", [(r["id"],) for r in rows]
                )
        # Cascades to chunks, symbols and edges; edges in *other* files that
        # pointed at these symbols get target_symbol = NULL (ON DELETE SET NULL).
        conn.execute("DELETE FROM files WHERE path = ?", (path,))

    def _insert_file(
        self,
        conn: sqlite3.Connection,
        file: FileRecord,
        chunks: Sequence[ChunkRecord],
        blobs: Sequence[bytes],
    ) -> list[int]:
        now = int(time.time())
        conn.execute(
            "INSERT INTO files(path, content_hash, mtime_ns, size, language, tags, indexed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (file.path, file.content_hash, file.mtime_ns, file.size, file.language, file.tags, now),
        )
        chunk_ids: list[int] = []
        for chunk, blob in zip(chunks, blobs, strict=True):
            cursor = conn.execute(
                "INSERT INTO chunks(file_path, chunk_hash, kind, symbol, parent_symbol, "
                "start_line, end_line, content, embed_text, symbol_terms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    file.path,
                    chunk.chunk_hash,
                    chunk.kind,
                    chunk.symbol,
                    chunk.parent_symbol,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.content,
                    chunk.embed_text,
                    chunk.symbol_terms,
                ),
            )
            chunk_id = _rowid(cursor)
            chunk_ids.append(chunk_id)
            conn.execute(
                "INSERT INTO chunks_fts(rowid, symbol_terms, content) VALUES (?, ?, ?)",
                (chunk_id, chunk.symbol_terms, chunk.content),
            )
            conn.execute(
                "INSERT OR IGNORE INTO embedding_cache(chunk_hash, embedder_id, embedding, "
                "created_at) VALUES (?, ?, ?, ?)",
                (chunk.chunk_hash, self._embedder_id, blob, now),
            )
            if self._use_vec0:
                conn.execute(
                    "INSERT INTO chunk_vecs(rowid, embedding) VALUES (?, ?)", (chunk_id, blob)
                )
        return chunk_ids

    def _insert_graph(
        self,
        conn: sqlite3.Connection,
        path: str,
        chunk_ids: Sequence[int],
        symbols: Sequence[SymbolRecord],
        edges: Sequence[EdgeRecord],
    ) -> None:
        symbol_ids: list[int] = []
        for s in symbols:
            chunk_id = chunk_ids[s.chunk_index] if s.chunk_index is not None else None
            cursor = conn.execute(
                "INSERT INTO symbols(file_path, chunk_id, name, kind, start_line, end_line) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (path, chunk_id, s.name, s.kind, s.start_line, s.end_line),
            )
            symbol_ids.append(_rowid(cursor))
        conn.executemany(
            "INSERT INTO edges(source_file, source_symbol, kind, target_name) VALUES (?, ?, ?, ?)",
            [(path, symbol_ids[e.source_index], e.kind, e.target_name) for e in edges],
        )

    # ------------------------------------------------------------- read path

    def manifest(self) -> dict[str, FileState]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT path, content_hash, mtime_ns, size FROM files"
            ).fetchall()
        return {r["path"]: FileState(r["content_hash"], r["mtime_ns"], r["size"]) for r in rows}

    def get_cached_embeddings(self, chunk_hashes: Sequence[str]) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        unique = list(dict.fromkeys(chunk_hashes))
        with self._lock:
            for i in range(0, len(unique), _SQL_BATCH):
                batch = unique[i : i + _SQL_BATCH]
                marks = ",".join("?" * len(batch))
                rows = self._conn.execute(
                    f"SELECT chunk_hash, embedding FROM embedding_cache "
                    f"WHERE embedder_id = ? AND chunk_hash IN ({marks})",
                    (self._embedder_id, *batch),
                ).fetchall()
                for r in rows:
                    found[r["chunk_hash"]] = list(_unpack(r["embedding"], self._dim))
        return found

    def vector_search(self, embedding: Sequence[float], k: int) -> list[Hit]:
        query = _pack(embedding, self._dim)
        with self._lock:
            if self._use_vec0:
                rows = self._conn.execute(
                    "SELECT rowid, distance FROM chunk_vecs "
                    "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                    (query, k),
                ).fetchall()
                # Unit vectors: L2 distance d relates to cosine as cos = 1 - d^2 / 2.
                return [Hit(r["rowid"], 1.0 - (r["distance"] ** 2) / 2.0) for r in rows]
            return self._scan_vectors(_unpack(query, self._dim), k)

    def _scan_vectors(self, query: tuple[float, ...], k: int) -> list[Hit]:
        cursor = self._conn.execute(
            "SELECT c.id AS id, ec.embedding AS embedding FROM chunks c "
            "JOIN embedding_cache ec ON ec.chunk_hash = c.chunk_hash AND ec.embedder_id = ?",
            (self._embedder_id,),
        )
        scored = (
            (sum(map(mul, query, _unpack(row["embedding"], self._dim))), row["id"])
            for row in cursor
        )
        return [Hit(chunk_id, score) for score, chunk_id in heapq.nlargest(k, scored)]

    def keyword_search(self, query: str, k: int, *, match_all: bool = False) -> list[Hit]:
        match = _fts_query(query, match_all)
        if not match:
            return []
        with self._lock:
            # Weights follow column order: symbol_terms counts 5x the body text.
            rows = self._conn.execute(
                "SELECT rowid, bm25(chunks_fts, 5.0, 1.0) AS rank FROM chunks_fts "
                "WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
                (match, k),
            ).fetchall()
        # FTS5's bm25() is lower-is-better (negative); flip so higher is better.
        return [Hit(r["rowid"], -r["rank"]) for r in rows]

    def get_chunks(self, chunk_ids: Sequence[int]) -> list[StoredChunk]:
        by_id: dict[int, StoredChunk] = {}
        unique = list(dict.fromkeys(chunk_ids))
        with self._lock:
            for i in range(0, len(unique), _SQL_BATCH):
                batch = unique[i : i + _SQL_BATCH]
                marks = ",".join("?" * len(batch))
                rows = self._conn.execute(
                    "SELECT id, file_path, kind, symbol, parent_symbol, start_line, end_line, "
                    f"content FROM chunks WHERE id IN ({marks})",
                    batch,
                ).fetchall()
                for r in rows:
                    by_id[r["id"]] = StoredChunk(
                        r["id"],
                        r["file_path"],
                        r["kind"],
                        r["symbol"],
                        r["parent_symbol"],
                        r["start_line"],
                        r["end_line"],
                        r["content"],
                    )
        return [by_id[i] for i in chunk_ids if i in by_id]

    # ----------------------------------------------------------------- graph

    def get_callers(self, symbol: str, limit: int = 50) -> list[SymbolRef]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT s.id, s.name, s.kind, s.file_path, s.start_line, s.end_line "
                "FROM edges e JOIN symbols s ON s.id = e.source_symbol "
                "WHERE e.kind = 'calls' AND e.target_name = ? "
                "ORDER BY s.file_path, s.start_line LIMIT ?",
                (symbol, limit),
            ).fetchall()
        return [SymbolRef(*tuple(r)) for r in rows]

    def get_callees(self, symbol: str, limit: int = 50) -> list[CalleeRef]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT e.target_name, e.kind AS edge_kind, "
                "t.id AS tid, t.name AS tname, t.kind AS tkind, t.file_path AS tfile, "
                "t.start_line AS tstart, t.end_line AS tend "
                "FROM symbols s "
                "JOIN edges e ON e.source_symbol = s.id "
                "LEFT JOIN symbols t ON t.id = e.target_symbol "
                "WHERE s.name = ? AND e.kind = 'calls' ORDER BY e.id LIMIT ?",
                (symbol, limit),
            ).fetchall()
        return [
            CalleeRef(
                r["target_name"],
                r["edge_kind"],
                None
                if r["tid"] is None
                else SymbolRef(
                    r["tid"], r["tname"], r["tkind"], r["tfile"], r["tstart"], r["tend"]
                ),
            )
            for r in rows
        ]

    def resolve_edges(self) -> int:
        """Link edges whose target name maps to exactly one symbol.

        Ambiguous names (two functions called `validate`) stay unresolved; that
        is the known v1 limitation until a type-aware resolver exists.
        """
        with self._transaction() as conn:
            cursor = conn.execute(
                "UPDATE edges SET target_symbol = "
                "  (SELECT s.id FROM symbols s WHERE s.name = edges.target_name) "
                "WHERE target_symbol IS NULL "
                "AND (SELECT COUNT(*) FROM symbols s WHERE s.name = edges.target_name) = 1"
            )
            return cursor.rowcount

    # ------------------------------------------------------- embedder changes

    def reset_embedder(self, embedder_id: str, embed_dim: int) -> int:
        """Switch to a different embedder. Returns how many chunks still lack a vector.

        Cached vectors for other embedders are kept (switching back is free).
        The caller re-embeds the chunks reported missing.
        """
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        with self._lock:
            with self._transaction() as conn:
                for key, value in (("embedder_id", embedder_id), ("embed_dim", str(embed_dim))):
                    conn.execute(
                        "INSERT INTO meta(key, value) VALUES (?, ?) "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                        (key, value),
                    )
            self._embedder_id = embedder_id
            self._dim = embed_dim
            if self._use_vec0:
                self._rebuild_vec_index()
            row = self._conn.execute(
                "SELECT COUNT(*) FROM chunks c WHERE NOT EXISTS ("
                "  SELECT 1 FROM embedding_cache ec "
                "  WHERE ec.chunk_hash = c.chunk_hash AND ec.embedder_id = ?)",
                (embedder_id,),
            ).fetchone()
        return int(row[0])


def _schema_sql() -> str:
    return resources.files("kibble_engine.store.sqlite").joinpath("schema.sql").read_text("utf-8")
