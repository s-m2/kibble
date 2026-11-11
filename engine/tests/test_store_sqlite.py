"""Tests for the SQLite store.

Every behavioural test runs against both vector backends. The sqlite-vec backend is
skipped automatically where the extension can't be loaded.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from kibble_engine.hashing import sha256_hex
from kibble_engine.store.models import (
    ChunkRecord,
    EdgeRecord,
    FileRecord,
    Hit,
    SymbolRecord,
)
from kibble_engine.store.sqlite import (
    SCHEMA_VERSION,
    EmbedderMismatchError,
    SchemaVersionError,
    SqliteStore,
)

DIM = 4
EMBEDDER = "test-embedder"

X = [1.0, 0.0, 0.0, 0.0]
Y = [0.0, 1.0, 0.0, 0.0]
Z = [0.0, 0.0, 1.0, 0.0]
XY = [1.0, 1.0, 0.0, 0.0]  # between X and Y


def _vec0_available() -> bool:
    try:
        import sqlite_vec
    except ImportError:
        return False
    conn = sqlite3.connect(":memory:")
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
    except (AttributeError, sqlite3.Error):
        return False
    finally:
        conn.close()
    return True


def chunk(
    text: str,
    vec: list[float],
    *,
    symbol: str | None = None,
    terms: str | None = None,
    lines: tuple[int, int] = (1, 5),
) -> ChunkRecord:
    return ChunkRecord(
        chunk_hash=sha256_hex(text),
        kind="function",
        start_line=lines[0],
        end_line=lines[1],
        content=text,
        embed_text=f"embed:{text}",
        embedding=vec,
        symbol=symbol,
        symbol_terms=terms if terms is not None else (symbol or ""),
    )


def file(path: str, content_hash: str = "h1") -> FileRecord:
    return FileRecord(path=path, content_hash=content_hash, mtime_ns=1, size=10, language="ts")


@pytest.fixture(params=["python", "vec0"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[SqliteStore]:
    backend = request.param
    if backend == "vec0" and not _vec0_available():
        pytest.skip("sqlite-vec extension not loadable here")
    s = SqliteStore.open(
        tmp_path / "index.db", embedder_id=EMBEDDER, embed_dim=DIM, vector_backend=backend
    )
    yield s
    s.close()


def ids(hits: list[Hit]) -> list[int]:
    return [h.chunk_id for h in hits]


# --------------------------------------------------------------------- basics


def test_fresh_store_is_empty(store: SqliteStore) -> None:
    assert store.counts() == {
        "files": 0,
        "chunks": 0,
        "symbols": 0,
        "edges": 0,
        "embedding_cache": 0,
    }
    assert store.manifest() == {}
    assert store.get_meta("embedder_id") == EMBEDDER
    assert store.get_meta("embed_dim") == str(DIM)


def test_replace_file_makes_content_searchable(store: SqliteStore) -> None:
    store.replace_file(
        file("auth.ts"),
        [
            chunk("function validateToken(t) {}", X, symbol="validateToken"),
            chunk("function login(u) {}", Y, symbol="login"),
        ],
    )
    assert store.manifest()["auth.ts"].content_hash == "h1"

    kw = store.keyword_search("validateToken", 5)
    assert [c.symbol for c in store.get_chunks(ids(kw))] == ["validateToken"]

    vec = store.vector_search(X, 5)
    assert [c.symbol for c in store.get_chunks(ids(vec))][0] == "validateToken"
    assert vec[0].score > vec[1].score


def test_vector_search_ranks_by_cosine(store: SqliteStore) -> None:
    store.replace_file(
        file("a.ts"),
        [chunk("x", X, symbol="x"), chunk("y", Y, symbol="y"), chunk("z", Z, symbol="z")],
    )
    hits = store.vector_search(XY, 3)
    symbols = [c.symbol for c in store.get_chunks(ids(hits))]
    assert set(symbols[:2]) == {"x", "y"}
    assert symbols[2] == "z"


def test_replace_file_replaces_previous_content(store: SqliteStore) -> None:
    store.replace_file(file("a.ts", "h1"), [chunk("old body alpha", X, symbol="alpha")])
    old_ids = ids(store.keyword_search("alpha", 5))
    assert old_ids

    store.replace_file(file("a.ts", "h2"), [chunk("new body beta", Y, symbol="beta")])

    assert store.manifest()["a.ts"].content_hash == "h2"
    assert store.keyword_search("alpha", 5) == []  # stale FTS row is gone
    assert [c.symbol for c in store.get_chunks(ids(store.keyword_search("beta", 5)))] == ["beta"]
    # Stale vectors are gone too: only the new chunk can be returned.
    vec_ids = ids(store.vector_search(X, 10))
    assert vec_ids and not set(vec_ids) & set(old_ids)
    assert store.counts()["chunks"] == 1


def test_remove_file_deletes_everything(store: SqliteStore) -> None:
    store.replace_file(
        file("a.ts"),
        [chunk("body", X, symbol="fn")],
        [SymbolRecord("fn", "function", 1, 5, 0)],
        [EdgeRecord(0, "calls", "other")],
    )
    store.remove_file("a.ts")
    counts = store.counts()
    assert counts["files"] == counts["chunks"] == counts["symbols"] == counts["edges"] == 0
    assert store.keyword_search("fn", 5) == []
    assert store.vector_search(X, 5) == []


def test_data_survives_reopen(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    with SqliteStore.open(db, embedder_id=EMBEDDER, embed_dim=DIM) as s:
        s.replace_file(file("a.ts"), [chunk("persisted body", X, symbol="persisted")])
    with SqliteStore.open(db, embedder_id=EMBEDDER, embed_dim=DIM) as s:
        assert "a.ts" in s.manifest()
        assert len(s.keyword_search("persisted", 5)) == 1
        assert len(s.vector_search(X, 5)) == 1


# ------------------------------------------------------------------ atomicity


def test_failure_mid_replace_rolls_everything_back(
    store: SqliteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core promise: a crash between steps leaves the old file fully intact."""
    store.replace_file(
        file("a.ts", "h1"),
        [chunk("original body", X, symbol="original")],
        [SymbolRecord("original", "function", 1, 5, 0)],
        [EdgeRecord(0, "calls", "helper")],
    )
    before = store.counts()
    before_vec = ids(store.vector_search(X, 10))

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("simulated crash after chunks were written")

    monkeypatch.setattr(store, "_insert_graph", boom)
    with pytest.raises(RuntimeError, match="simulated crash"):
        store.replace_file(
            file("a.ts", "h2"),
            [chunk("replacement body", Y, symbol="replacement")],
            [SymbolRecord("replacement", "function", 1, 5, 0)],
        )

    assert store.manifest()["a.ts"].content_hash == "h1"
    assert store.counts() == before
    assert [c.symbol for c in store.get_chunks(ids(store.keyword_search("original", 5)))] == [
        "original"
    ]
    assert store.keyword_search("replacement", 5) == []
    assert ids(store.vector_search(X, 10)) == before_vec
    assert [c.target_name for c in store.get_callees("original")] == ["helper"]


def test_invalid_input_is_rejected_before_any_write(store: SqliteStore) -> None:
    store.replace_file(file("a.ts", "h1"), [chunk("original body", X, symbol="original")])
    before = store.counts()

    with pytest.raises(ValueError, match="dimensions"):
        store.replace_file(file("a.ts", "h2"), [chunk("bad", [1.0, 0.0], symbol="bad")])
    with pytest.raises(ValueError, match="repo-relative"):
        store.replace_file(file("/abs/path.ts"), [chunk("x", X)])
    with pytest.raises(ValueError, match="source_index"):
        store.replace_file(
            file("a.ts", "h3"),
            [chunk("x", X)],
            [SymbolRecord("s", "function", 1, 2)],
            [EdgeRecord(5, "calls", "t")],
        )
    with pytest.raises(ValueError, match="non-zero"):
        store.replace_file(file("a.ts", "h4"), [chunk("x", [0.0] * DIM)])

    assert store.manifest()["a.ts"].content_hash == "h1"
    assert store.counts() == before


# -------------------------------------------------------------------- caching


def test_embedding_cache_returns_normalised_vectors(store: SqliteStore) -> None:
    c = chunk("body", [2.0, 0.0, 0.0, 0.0], symbol="s")
    store.replace_file(file("a.ts"), [c])
    got = store.get_cached_embeddings([c.chunk_hash, "unknown-hash"])
    assert set(got) == {c.chunk_hash}
    assert got[c.chunk_hash] == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_cache_survives_file_replacement(store: SqliteStore) -> None:
    c = chunk("kept body", X, symbol="kept")
    store.replace_file(file("a.ts", "h1"), [c])
    store.replace_file(file("a.ts", "h2"), [chunk("different", Y, symbol="d")])
    # The old chunk's embedding is still cached: reverting the edit is free.
    assert c.chunk_hash in store.get_cached_embeddings([c.chunk_hash])


# ------------------------------------------------------------------- keywords


@pytest.mark.parametrize(
    "query",
    ["auth.validate", "user-id", 'say "hello"', "a:b", "foo*", "(", "NEAR/2 x", "  ", "---"],
)
def test_keyword_search_never_raises_on_punctuation(store: SqliteStore, query: str) -> None:
    store.replace_file(
        file("a.ts"), [chunk("auth validate user id hello", X, symbol="s", terms="auth validate")]
    )
    store.keyword_search(query, 5)  # must not raise


def test_keyword_search_finds_dotted_identifier(store: SqliteStore) -> None:
    store.replace_file(file("a.ts"), [chunk("call auth.validate(token)", X, symbol="s")])
    assert len(store.keyword_search("auth.validate", 5)) == 1


def test_symbol_terms_outrank_body_matches(store: SqliteStore) -> None:
    store.replace_file(
        file("a.ts"),
        [
            chunk("mentions token somewhere in a long body of text", X, symbol="a", terms="a"),
            chunk("tiny", Y, symbol="b", terms="token"),
        ],
    )
    hits = store.keyword_search("token", 5)
    assert [c.symbol for c in store.get_chunks(ids(hits))][0] == "b"


# ---------------------------------------------------------------------- graph


def test_callers_callees_and_resolution(store: SqliteStore) -> None:
    store.replace_file(
        file("util.ts"),
        [chunk("function validate() {}", X, symbol="validate")],
        [SymbolRecord("validate", "function", 1, 5, 0)],
    )
    store.replace_file(
        file("auth.ts"),
        [chunk("function login() { validate() }", Y, symbol="login")],
        [SymbolRecord("login", "function", 1, 5, 0)],
        [EdgeRecord(0, "calls", "validate"), EdgeRecord(0, "calls", "missing")],
    )

    assert [s.name for s in store.get_callers("validate")] == ["login"]
    callees = store.get_callees("login")
    assert [c.target_name for c in callees] == ["validate", "missing"]
    assert all(c.target is None for c in callees)  # not resolved yet

    assert store.resolve_edges() == 1  # "missing" has no symbol
    callees = store.get_callees("login")
    assert callees[0].target is not None and callees[0].target.file_path == "util.ts"
    assert callees[1].target is None


def test_ambiguous_names_stay_unresolved(store: SqliteStore) -> None:
    for path in ("a.ts", "b.ts"):
        store.replace_file(
            file(path),
            [chunk(f"function dup() {{}} // {path}", X, symbol="dup")],
            [SymbolRecord("dup", "function", 1, 5, 0)],
        )
    store.replace_file(
        file("c.ts"),
        [chunk("function user() { dup() }", Y, symbol="user")],
        [SymbolRecord("user", "function", 1, 5, 0)],
        [EdgeRecord(0, "calls", "dup")],
    )
    assert store.resolve_edges() == 0
    assert store.get_callees("user")[0].target is None


def test_replacing_a_file_replaces_its_outgoing_edges(store: SqliteStore) -> None:
    store.replace_file(
        file("auth.ts", "h1"),
        [chunk("function login() { old() }", Y, symbol="login")],
        [SymbolRecord("login", "function", 1, 5, 0)],
        [EdgeRecord(0, "calls", "old")],
    )
    store.replace_file(
        file("auth.ts", "h2"),
        [chunk("function login() { fresh() }", Y, symbol="login")],
        [SymbolRecord("login", "function", 1, 5, 0)],
        [EdgeRecord(0, "calls", "fresh")],
    )
    assert store.get_callers("old") == []
    assert [s.name for s in store.get_callers("fresh")] == ["login"]
    assert store.counts()["edges"] == 1


def test_replacing_a_target_file_unlinks_incoming_edges(store: SqliteStore) -> None:
    store.replace_file(
        file("util.ts", "h1"),
        [chunk("function validate() {}", X, symbol="validate")],
        [SymbolRecord("validate", "function", 1, 5, 0)],
    )
    store.replace_file(
        file("auth.ts"),
        [chunk("function login() { validate() }", Y, symbol="login")],
        [SymbolRecord("login", "function", 1, 5, 0)],
        [EdgeRecord(0, "calls", "validate")],
    )
    store.resolve_edges()
    assert store.get_callees("login")[0].target is not None

    # util.ts changes: its symbol ids change, so the old link must not dangle.
    store.replace_file(
        file("util.ts", "h2"),
        [chunk("function validate() { /* v2 */ }", X, symbol="validate")],
        [SymbolRecord("validate", "function", 1, 5, 0)],
    )
    assert store.get_callees("login")[0].target is None
    store.resolve_edges()
    assert store.get_callees("login")[0].target is not None  # re-linked


# ------------------------------------------------------------ embedder changes


def test_reopening_with_a_different_embedder_is_refused(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    SqliteStore.open(db, embedder_id=EMBEDDER, embed_dim=DIM).close()
    with pytest.raises(EmbedderMismatchError, match="reset_embedder"):
        SqliteStore.open(db, embedder_id="other", embed_dim=DIM)
    with pytest.raises(EmbedderMismatchError):
        SqliteStore.open(db, embedder_id=EMBEDDER, embed_dim=DIM + 1)


def test_reset_embedder_reports_chunks_needing_vectors(store: SqliteStore) -> None:
    store.replace_file(file("a.ts"), [chunk("one", X, symbol="one"), chunk("two", Y, symbol="two")])

    missing = store.reset_embedder("new-embedder", 3)
    assert missing == 2  # nothing cached for the new embedder yet
    assert store.vector_search([1.0, 0.0, 0.0], 5) == []

    # Switching back is free: the old vectors were never deleted.
    assert store.reset_embedder(EMBEDDER, DIM) == 0
    assert len(store.vector_search(X, 5)) == 2


def test_unknown_schema_version_is_refused(tmp_path: Path) -> None:
    db = tmp_path / "index.db"
    SqliteStore.open(db, embedder_id=EMBEDDER, embed_dim=DIM).close()
    raw = sqlite3.connect(db)
    raw.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    raw.close()
    with pytest.raises(SchemaVersionError, match="delete the .kibble"):
        SqliteStore.open(db, embedder_id=EMBEDDER, embed_dim=DIM)


# ------------------------------------------------------------------- backends


def test_backends_rank_identically(tmp_path: Path) -> None:
    if not _vec0_available():
        pytest.skip("sqlite-vec extension not loadable here")
    chunks = [chunk(f"c{i}", v, symbol=f"c{i}") for i, v in enumerate([X, Y, Z, XY])]
    results = []
    for backend in ("python", "vec0"):
        with SqliteStore.open(
            tmp_path / f"{backend}.db", embedder_id=EMBEDDER, embed_dim=DIM, vector_backend=backend
        ) as s:
            s.replace_file(file("a.ts"), chunks)
            results.append([c.symbol for c in s.get_chunks(ids(s.vector_search([3, 2, 0, 0], 4)))])
    assert results[0] == results[1]


def test_vec_index_is_rebuilt_after_python_mode_writes(tmp_path: Path) -> None:
    """Writes made without the extension must still be visible once it's back."""
    if not _vec0_available():
        pytest.skip("sqlite-vec extension not loadable here")
    db = tmp_path / "index.db"
    with SqliteStore.open(db, embedder_id=EMBEDDER, embed_dim=DIM, vector_backend="vec0") as s:
        s.replace_file(file("a.ts"), [chunk("one", X, symbol="one")])
    with SqliteStore.open(db, embedder_id=EMBEDDER, embed_dim=DIM, vector_backend="python") as s:
        s.replace_file(file("b.ts"), [chunk("two", Y, symbol="two")])
    with SqliteStore.open(db, embedder_id=EMBEDDER, embed_dim=DIM, vector_backend="vec0") as s:
        symbols = [c.symbol for c in s.get_chunks(ids(s.vector_search(Y, 5)))]
        assert symbols[0] == "two" and set(symbols) == {"one", "two"}
