from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from kibble_engine.core import sweep as sweep_module
from kibble_engine.core.scanner import scan_repo
from kibble_engine.core.sweep import SweepResult, sweep
from kibble_engine.hashing import sha256_bytes_hex
from kibble_engine.store.models import FileState
from kibble_engine.testing.helpers import init_repo, write


def state_of(root: Path, rel: str, *, indexed_at: float | None = None) -> FileState:
    """The manifest entry the indexer would have recorded for `rel` right now."""
    path = root / rel
    stat = path.stat()
    stamp = int(indexed_at if indexed_at is not None else time.time())
    return FileState(sha256_bytes_hex(path.read_bytes()), stat.st_mtime_ns, stat.st_size, stamp)


def run(root: Path, manifest: dict[str, FileState], only: list[str] | None = None) -> SweepResult:
    return sweep(root, manifest, scan_repo(root, use_git=False), only=only)


def test_new_and_deleted_files(tmp_path: Path) -> None:
    init_repo(tmp_path, {"a.txt": "alpha\n", "b.txt": "beta\n"}, git=False)
    manifest = {"a.txt": state_of(tmp_path, "a.txt"), "gone.txt": FileState("h", 1, 1, 1)}
    result = run(tmp_path, manifest)
    assert [f.path for f in result.new] == ["b.txt"]
    assert result.deleted == ["gone.txt"]
    assert result.unchanged == 1 and not result.changed and not result.touched


def test_unchanged_files_are_never_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    init_repo(tmp_path, {"a.txt": "alpha\n", "b.txt": "beta\n"}, git=False)  # old mtimes
    manifest = {p: state_of(tmp_path, p) for p in ("a.txt", "b.txt")}

    def forbidden(path: Path) -> str:
        raise AssertionError(f"fast path must not read {path}")

    monkeypatch.setattr(sweep_module, "_hash_file", forbidden)
    result = run(tmp_path, manifest)
    assert result.unchanged == 2


def test_content_change_is_detected(tmp_path: Path) -> None:
    init_repo(tmp_path, {"a.txt": "alpha\n"}, git=False)
    manifest = {"a.txt": state_of(tmp_path, "a.txt")}
    write(tmp_path, "a.txt", "different content\n")
    result = run(tmp_path, manifest)
    assert [f.path for f in result.changed] == ["a.txt"]


def test_mtime_bump_with_identical_content_is_only_a_touch(tmp_path: Path) -> None:
    init_repo(tmp_path, {"a.txt": "alpha\n"}, git=False)
    manifest = {"a.txt": state_of(tmp_path, "a.txt")}
    os.utime(tmp_path / "a.txt", (time.time() - 60, time.time() - 60))  # e.g. a checkout
    result = run(tmp_path, manifest)
    assert [f.path for f in result.touched] == ["a.txt"]
    assert not result.changed and not result.new


def test_racy_edit_with_unchanged_size_and_mtime_is_still_caught(tmp_path: Path) -> None:
    """Edited again within the timestamp granularity: stat looks identical, content isn't."""
    init_repo(tmp_path, {"a.txt": "alpha\n"}, git=False, age_s=0)  # mtime ~ now
    manifest = {"a.txt": state_of(tmp_path, "a.txt")}  # indexed_at ~ now: the racy window
    before = (tmp_path / "a.txt").stat()
    (tmp_path / "a.txt").write_text("omega\n")  # same length
    os.utime(tmp_path / "a.txt", ns=(before.st_atime_ns, before.st_mtime_ns))  # identical stat
    result = run(tmp_path, manifest)
    assert [f.path for f in result.changed] == ["a.txt"]


def test_old_files_stay_on_the_fast_path_even_if_the_index_is_recent(tmp_path: Path) -> None:
    init_repo(tmp_path, {"a.txt": "alpha\n"}, git=False)  # mtime an hour ago
    manifest = {"a.txt": state_of(tmp_path, "a.txt", indexed_at=time.time())}
    assert run(tmp_path, manifest).unchanged == 1


def test_file_that_vanishes_during_the_sweep_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_repo(tmp_path, {"a.txt": "alpha\n"}, git=False)
    manifest = {"a.txt": state_of(tmp_path, "a.txt")}
    os.utime(tmp_path / "a.txt", (time.time() - 60, time.time() - 60))
    monkeypatch.setattr(sweep_module, "_hash_file", lambda path: None)
    result = run(tmp_path, manifest)
    assert not (result.new or result.changed or result.touched or result.deleted)


def test_targeted_sweeps_only_conclude_deletions_for_their_own_paths(tmp_path: Path) -> None:
    init_repo(tmp_path, {"a.txt": "alpha\n"}, git=False)
    manifest = {
        "a.txt": state_of(tmp_path, "a.txt"),
        "other.txt": FileState("h", 1, 1, 1),  # not in the scan, but not asked about
        "asked.txt": FileState("h", 1, 1, 1),
    }
    result = run(tmp_path, manifest, only=["a.txt", "asked.txt"])
    assert result.deleted == ["asked.txt"]


def test_dirty_is_new_plus_changed_in_path_order(tmp_path: Path) -> None:
    init_repo(tmp_path, {"z.txt": "z\n", "a.txt": "a\n", "m.txt": "m\n"}, git=False)
    manifest = {"m.txt": FileState("stale-hash", 1, 1, 1)}
    assert [f.path for f in run(tmp_path, manifest).dirty] == ["a.txt", "m.txt", "z.txt"]
