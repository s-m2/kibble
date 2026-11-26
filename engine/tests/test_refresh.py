"""Freshness scenarios: the index must follow the working tree, and say so when it can't."""

from __future__ import annotations

import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from kibble_engine.core import sweep as sweep_module
from kibble_engine.core.refresh import Refresher
from kibble_engine.embedders import OpenAICompatibleEmbedder
from kibble_engine.store.sqlite import SqliteStore
from kibble_engine.testing.fake_embedding_server import FakeEmbeddingServer
from kibble_engine.testing.helpers import (
    git_available,
    init_repo,
    make_embedder,
    open_store,
    write,
)

Env = tuple[FakeEmbeddingServer, OpenAICompatibleEmbedder, SqliteStore, Refresher]
MODES = [False, True]  # use git for file discovery?


def make(tmp_path: Path, files: Mapping[str, str | bytes], *, git: bool = False, **kw: Any) -> Env:
    if git and not git_available():
        pytest.skip("git not installed")
    init_repo(tmp_path, files, git=git)
    server = FakeEmbeddingServer(dim=32)
    embedder = make_embedder(server)
    store = open_store(tmp_path, embedder)
    return server, embedder, store, Refresher(tmp_path, store, embedder, use_git=git, **kw)


def found(store: SqliteStore, term: str) -> bool:
    return bool(store.keyword_search(term, 5))


FILES = {"a.txt": "alphaterm one\n", "b.txt": "betaterm two\n", "c.txt": "gammaterm three\n"}


@pytest.mark.parametrize("git", MODES)
def test_first_refresh_indexes_everything_and_the_next_is_free(
    tmp_path: Path, git: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, embedder, store, refresher = make(tmp_path, FILES, git=git)
    first = refresher.refresh()
    assert (first.new, len(first.refreshed), first.stale) == (3, 3, False)

    requests_before = embedder.stats.requests
    monkeypatch.setattr(sweep_module, "_hash_file", lambda p: pytest.fail("read on fast path"))
    second = refresher.refresh()
    assert second.unchanged == 3 and not second.refreshed
    assert embedder.stats.requests == requests_before  # no embedding calls at all


@pytest.mark.parametrize("git", MODES)
def test_edit_then_query_sees_the_edit_immediately(tmp_path: Path, git: bool) -> None:
    server, embedder, store, refresher = make(tmp_path, FILES, git=git)
    refresher.refresh()
    assert found(store, "alphaterm")

    write(tmp_path, "a.txt", "replacementterm entirely new\n")
    report = refresher.ensure_fresh()  # what a query does first, with no watcher or hook

    assert report.refreshed == ["a.txt"] and not report.stale
    assert found(store, "replacementterm") and not found(store, "alphaterm")


@pytest.mark.parametrize("git", MODES)
def test_new_file_then_query_finds_it_immediately(tmp_path: Path, git: bool) -> None:
    server, embedder, store, refresher = make(tmp_path, FILES, git=git)
    refresher.refresh()
    write(tmp_path, "new.txt", "freshlycreatedterm here\n")
    refresher.ensure_fresh()
    assert found(store, "freshlycreatedterm")


@pytest.mark.parametrize("git", MODES)
def test_deleted_file_disappears_from_results(tmp_path: Path, git: bool) -> None:
    server, embedder, store, refresher = make(tmp_path, FILES, git=git)
    refresher.refresh()
    (tmp_path / "b.txt").unlink()
    report = refresher.ensure_fresh()
    assert report.removed == ["b.txt"] and not found(store, "betaterm")
    assert "b.txt" not in store.manifest()


def test_same_size_edit_with_restored_mtime_is_still_caught(tmp_path: Path) -> None:
    server, embedder, store, refresher = make(tmp_path, {"a.txt": "alpha\n"})
    os.utime(tmp_path / "a.txt")  # mtime = now: inside the racy window of the index time
    refresher.refresh()
    before = (tmp_path / "a.txt").stat()
    (tmp_path / "a.txt").write_text("omega\n")
    os.utime(tmp_path / "a.txt", ns=(before.st_atime_ns, before.st_mtime_ns))
    report = refresher.refresh()
    assert report.refreshed == ["a.txt"] and found(store, "omega")


def test_a_touch_without_content_change_costs_no_embedding(tmp_path: Path) -> None:
    server, embedder, store, refresher = make(tmp_path, FILES)
    refresher.refresh()
    before = embedder.stats.requests
    os.utime(tmp_path / "a.txt")  # e.g. a branch switch rewrote identical content
    report = refresher.refresh()
    assert report.touched == 1 and not report.refreshed
    assert embedder.stats.requests == before


def test_budget_defers_the_rest_as_pending_until_done(tmp_path: Path) -> None:
    files = {f"f{i}.txt": f"termnumber{i}\n" for i in range(5)}
    server, embedder, store, refresher = make(tmp_path, files)
    first = refresher.refresh(max_files=2)
    assert len(first.refreshed) == 2 and len(first.pending) == 3 and first.stale
    second = refresher.refresh(max_files=2)
    assert len(second.refreshed) == 2 and len(second.pending) == 1 and second.stale
    third = refresher.refresh(max_files=2)
    assert len(third.refreshed) == 1 and not third.pending and not third.stale
    assert store.counts()["files"] == 5


def test_targeted_refresh_examines_only_the_given_paths(tmp_path: Path) -> None:
    server, embedder, store, refresher = make(tmp_path, FILES)
    refresher.refresh()
    write(tmp_path, "a.txt", "changed alpha\n")
    write(tmp_path, "b.txt", "changed beta\n")

    only_a = refresher.refresh(paths=["a.txt"])
    assert only_a.refreshed == ["a.txt"]
    assert found(store, "changed") and found(store, "betaterm")  # b still has its old entry
    assert [f.path for f in refresher.plan()[1].changed] == ["b.txt"]  # and is still dirty

    by_absolute_path = refresher.refresh(paths=[tmp_path / "b.txt"])
    assert by_absolute_path.refreshed == ["b.txt"]

    (tmp_path / "c.txt").unlink()
    removed = refresher.refresh(paths=["c.txt", tmp_path.parent / "outside.txt"])
    assert removed.removed == ["c.txt"] and removed.scanned == 0


def test_outage_keeps_old_entries_flags_stale_and_recovers(tmp_path: Path) -> None:
    server, embedder, store, refresher = make(tmp_path, FILES)
    refresher.refresh()
    write(tmp_path, "a.txt", "newversionterm\n")

    server.fail_next(503, times=10)
    report = refresher.ensure_fresh()
    assert report.stale and report.aborted and report.pending == ["a.txt"]
    assert report.freshness()["stale"] is True
    assert found(store, "alphaterm") and not found(store, "newversionterm")  # old entry intact

    server._script.clear()  # the endpoint recovers
    recovered = refresher.ensure_fresh()
    assert not recovered.stale and found(store, "newversionterm")


def test_a_file_that_outgrows_the_size_cap_leaves_the_index(tmp_path: Path) -> None:
    server, embedder, store, refresher = make(tmp_path, {"a.txt": "small\n"}, max_file_bytes=200)
    refresher.refresh()
    assert "a.txt" in store.manifest()
    write(tmp_path, "a.txt", "word " * 100)
    report = refresher.refresh()
    assert report.removed == ["a.txt"] and "a.txt" not in store.manifest()


def test_newly_ignored_files_leave_the_index(tmp_path: Path) -> None:
    server, embedder, store, refresher = make(tmp_path, FILES, git=True)
    refresher.refresh()
    write(tmp_path, ".gitignore", "a.txt\n")
    report = refresher.refresh()
    assert "a.txt" in report.removed and not found(store, "alphaterm")


def test_plan_has_no_side_effects(tmp_path: Path) -> None:
    server, embedder, store, refresher = make(tmp_path, FILES)
    refresher.refresh()
    write(tmp_path, "a.txt", "changed\n")
    write(tmp_path, "new.txt", "brand new\n")
    manifest_before, requests_before = store.manifest(), embedder.stats.requests

    _, plan = refresher.plan()

    assert [f.path for f in plan.changed] == ["a.txt"] and [f.path for f in plan.new] == ["new.txt"]
    assert store.manifest() == manifest_before and embedder.stats.requests == requests_before


def test_freshness_metadata_describes_the_outcome(tmp_path: Path) -> None:
    server, embedder, store, refresher = make(tmp_path, FILES)
    meta = refresher.refresh().freshness()
    assert meta["stale"] is False and meta["checked_files"] == 3 and meta["refreshed_files"] == 3
    assert set(meta) == {
        "stale",
        "checked_files",
        "refreshed_files",
        "removed_files",
        "pending_files",
        "failed_files",
        "aborted",
        "duration_ms",
    }


def test_concurrent_refreshes_do_not_duplicate_work(tmp_path: Path) -> None:
    files = {f"f{i}.txt": f"concurrentterm{i}\n" for i in range(8)}
    server, embedder, store, refresher = make(tmp_path, files)
    with ThreadPoolExecutor(max_workers=3) as pool:
        reports = list(pool.map(lambda _: refresher.refresh(), range(3)))
    assert sum(len(r.refreshed) for r in reports) == 8
    assert store.counts()["files"] == 8
