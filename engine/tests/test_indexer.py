from __future__ import annotations

import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from kibble_engine.core.indexer import Indexer
from kibble_engine.core.refresh import Refresher
from kibble_engine.core.scanner import ScannedFile, scan_repo
from kibble_engine.embedders import OpenAICompatibleEmbedder
from kibble_engine.store.sqlite import SqliteStore
from kibble_engine.testing.fake_embedding_server import FakeEmbeddingServer
from kibble_engine.testing.helpers import (
    grammars_available,
    init_repo,
    make_embedder,
    open_store,
    write,
)


def setup(
    tmp_path: Path, files: Mapping[str, str | bytes], **server_kwargs: Any
) -> tuple[FakeEmbeddingServer, OpenAICompatibleEmbedder, SqliteStore]:
    init_repo(tmp_path, files, git=False)
    server = FakeEmbeddingServer(dim=32, **server_kwargs)
    embedder = make_embedder(server)
    store = open_store(tmp_path, embedder)
    return server, embedder, store


def scanned(root: Path, *paths: str) -> list[ScannedFile]:
    found = scan_repo(root, use_git=False).files
    return [found[p] for p in paths] if paths else list(found.values())


def test_many_files_share_embedding_requests(tmp_path: Path) -> None:
    files = {f"f{i}.txt": f"unique words number {i} here\n" for i in range(12)}
    server, embedder, store = setup(tmp_path, files)
    before = len(server.requests)

    run = Indexer(tmp_path, store, embedder).index_files(scanned(tmp_path))

    assert len(run.outcomes) == 12 and not run.failed and not run.pending
    assert len(server.requests) - before == 1  # one request for all twelve files
    assert store.counts()["files"] == 12


def test_windows_bound_how_much_is_in_flight(tmp_path: Path) -> None:
    files = {f"f{i}.txt": f"words {i}\n" for i in range(9)}
    server, embedder, store = setup(tmp_path, files)
    before = len(server.requests)
    Indexer(tmp_path, store, embedder, window_files=4).index_files(scanned(tmp_path))
    assert len(server.requests) - before == 3  # windows of 4, 4 and 1 files


def test_reindexing_identical_chunks_uses_the_cache(tmp_path: Path) -> None:
    server, embedder, store = setup(tmp_path, {"a.txt": "stable content here\n"})
    indexer = Indexer(tmp_path, store, embedder)
    first = indexer.index_files(scanned(tmp_path))
    before = len(server.requests)
    second = indexer.index_files(scanned(tmp_path))

    assert first.embedded_chunks == 1 and first.cached_chunks == 0
    assert second.embedded_chunks == 0 and second.cached_chunks == 1
    assert len(server.requests) == before  # no embedding call at all


def test_symbols_are_stored_when_a_grammar_is_available(tmp_path: Path) -> None:
    if not grammars_available():
        pytest.skip("tree-sitter grammars not installed")
    source = "export function validateToken(t) {\n  return t.exp > Date.now();\n}\n"
    _, embedder, store = setup(tmp_path, {"auth.ts": source})
    Indexer(tmp_path, store, embedder).index_files(scanned(tmp_path))
    assert store.counts()["symbols"] >= 1


def test_binary_content_is_recorded_without_chunks_and_not_retried(tmp_path: Path) -> None:
    server, embedder, store = setup(
        tmp_path, {"data.xyz": b"header\x00\x01\x02binary", "ok.txt": "text here\n"}
    )
    run = Indexer(tmp_path, store, embedder).index_files(scanned(tmp_path))
    by_path = {o.path: o for o in run.outcomes}
    assert by_path["data.xyz"].status == "binary" and by_path["data.xyz"].chunks == 0
    assert "data.xyz" in store.manifest()  # remembered, so sweeps don't re-read it
    assert store.counts()["chunks"] == 1


def test_whitespace_only_file_is_recorded_as_empty(tmp_path: Path) -> None:
    _, embedder, store = setup(tmp_path, {"blank.txt": "   \n\n  \n"})
    run = Indexer(tmp_path, store, embedder).index_files(scanned(tmp_path))
    assert run.outcomes[0].status == "empty" and "blank.txt" in store.manifest()


def test_unreadable_file_fails_alone(tmp_path: Path) -> None:
    _, embedder, store = setup(tmp_path, {"a.txt": "alpha\n", "b.txt": "beta\n"})
    files = scanned(tmp_path)
    (tmp_path / "a.txt").unlink()  # vanishes after the scan
    run = Indexer(tmp_path, store, embedder).index_files(files)
    assert [p for p, _ in run.failed] == ["a.txt"] and "cannot read" in run.failed[0][1]
    assert [o.path for o in run.outcomes] == ["b.txt"]


def test_progress_reports_finished_files(tmp_path: Path) -> None:
    files = {f"f{i}.txt": f"words {i}\n" for i in range(5)}
    _, embedder, store = setup(tmp_path, files)
    seen: list[tuple[int, int]] = []
    Indexer(tmp_path, store, embedder, window_files=2).index_files(
        scanned(tmp_path), on_progress=lambda done, total: seen.append((done, total))
    )
    assert seen[-1] == (5, 5) and all(t == 5 for _, t in seen)
    assert [d for d, _ in seen] == sorted(d for d, _ in seen)


def test_an_edit_during_indexing_is_noticed_by_the_next_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stat is taken BEFORE the content is read, so a concurrent edit can never be
    recorded as if it had been indexed. The edit's timestamp is old enough that the
    racy-timestamp rule can't rescue a wrong ordering."""
    server, embedder, store = setup(tmp_path, {"a.txt": "original content\n"})
    target = tmp_path / "a.txt"
    files = scanned(tmp_path)
    real_read = Path.read_bytes

    def read_then_edit(self: Path) -> bytes:
        data = real_read(self)
        if self == target:
            target.write_text("edited meanwhile\n")  # the file changes right after we read it
            stamp = time.time() - 1800  # older than the index time, newer than the original
            os.utime(target, (stamp, stamp))
        return data

    monkeypatch.setattr(Path, "read_bytes", read_then_edit)
    Indexer(tmp_path, store, embedder).index_files(files)
    monkeypatch.setattr(Path, "read_bytes", real_read)

    assert len(store.keyword_search("original", 5)) == 1  # we indexed what we read
    report = Refresher(tmp_path, store, embedder, use_git=False).refresh()
    assert report.refreshed == ["a.txt"]  # ...and the edit is not lost
    assert len(store.keyword_search("meanwhile", 5)) == 1


# ------------------------------------------------------------------ failures


def test_endpoint_outage_stops_the_run_and_leaves_old_entries_intact(tmp_path: Path) -> None:
    server, embedder, store = setup(tmp_path, {"a.txt": "oldterm alpha\n", "b.txt": "beta\n"})
    indexer = Indexer(tmp_path, store, embedder)
    indexer.index_files(scanned(tmp_path))

    write(tmp_path, "a.txt", "newterm alpha changed\n")
    write(tmp_path, "c.txt", "gamma\n")
    server.fail_next(401, times=5, message="credits exhausted upstream")
    run = indexer.index_files(scanned(tmp_path, "a.txt", "c.txt"))

    assert run.aborted and "401" in run.aborted
    assert sorted(run.pending) == ["a.txt", "c.txt"] and not run.outcomes
    # a.txt still has its previous index entry: stale, but never half-written.
    assert len(store.keyword_search("oldterm", 5)) == 1
    assert store.keyword_search("newterm", 5) == []
    assert "c.txt" not in store.manifest()


def test_a_rejected_chunk_is_isolated_to_its_file(tmp_path: Path) -> None:
    big = " ".join(f"word{i}" for i in range(600))  # one ~2400-char chunk
    server, embedder, store = setup(
        tmp_path,
        {"ok1.txt": "hi\n", "big.txt": big, "ok2.txt": "yo\n"},
        max_input_chars=100,
    )
    embedder = make_embedder(server, max_input_chars=2000)
    run = Indexer(tmp_path, store, embedder).index_files(scanned(tmp_path))

    assert [p for p, _ in run.failed] == ["big.txt"]
    assert "embedding rejected" in run.failed[0][1]
    assert sorted(o.path for o in run.outcomes) == ["ok1.txt", "ok2.txt"]
    assert run.aborted is None


def test_pending_includes_files_not_yet_reached(tmp_path: Path) -> None:
    files = {f"f{i}.txt": f"words {i}\n" for i in range(6)}
    server, embedder, store = setup(tmp_path, files)
    indexer = Indexer(tmp_path, store, embedder, window_files=2)
    server.fail_next(401, times=10)
    run = indexer.index_files(scanned(tmp_path))
    assert run.aborted and sorted(run.pending) == sorted(files)
    assert store.counts()["files"] == 0
