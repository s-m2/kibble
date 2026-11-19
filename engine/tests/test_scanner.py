from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from kibble_engine.core.scanner import classify_tags, inspect_file, scan_paths, scan_repo
from kibble_engine.testing.helpers import git_available, init_repo

FILES = {
    "src/a.ts": "export const a = 1;\n",
    "tests/test_a.py": "def test(): pass\n",
    "logo.png": b"\x89PNG\r\n",
    "package-lock.json": "{}",
    "dist/bundle.min.js": "x",
    "empty.txt": "",
    "README.md": "# hi\n",
}


@pytest.mark.parametrize(
    ("path", "tags"),
    [
        ("src/a.ts", ""),
        ("tests/test_a.py", "test"),
        ("src/a.spec.ts", "test"),
        ("src/__tests__/x.js", "test"),
        ("node_modules/pkg/index.js", "vendor"),
        ("src/generated/api.ts", "generated"),
        ("proto/api_pb2.py", "generated"),
        ("vendor/lib/tests/t.py", "test vendor"),
    ],
)
def test_classify_tags(path: str, tags: str) -> None:
    assert classify_tags(path) == tags


def test_directory_walk_filters_and_prunes(tmp_path: Path) -> None:
    init_repo(tmp_path, {**FILES, "node_modules/x.js": "x", ".kibble/index.db": "x"}, git=False)
    result = scan_repo(tmp_path, use_git=False)
    assert not result.used_git
    assert sorted(result.files) == ["README.md", "src/a.ts", "tests/test_a.py"]
    assert result.skipped["logo.png"] == "binary extension"
    assert result.skipped["package-lock.json"] == "lockfile or generated bundle"
    assert result.skipped["empty.txt"] == "empty"
    assert result.files["tests/test_a.py"].tags == "test"


def test_git_mode_respects_gitignore_and_includes_untracked(tmp_path: Path) -> None:
    if not git_available():
        pytest.skip("git not installed")
    files = {**FILES, ".gitignore": "secret.txt\nbuild/\n", "secret.txt": "x", "build/out.js": "x"}
    init_repo(tmp_path, files)
    result = scan_repo(tmp_path)
    assert result.used_git
    assert "secret.txt" not in result.files and "build/out.js" not in result.files
    assert "src/a.ts" in result.files  # untracked but not ignored


def test_git_mode_skips_files_deleted_from_the_working_tree(tmp_path: Path) -> None:
    if not git_available():
        pytest.skip("git not installed")
    init_repo(tmp_path, {"keep.ts": "export {}\n", "gone.ts": "export {}\n"})
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    (tmp_path / "gone.ts").unlink()
    result = scan_repo(tmp_path)
    assert "keep.ts" in result.files and "gone.ts" not in result.files


def test_size_cap_and_symlinks(tmp_path: Path) -> None:
    init_repo(tmp_path, {"big.txt": "x" * 500, "small.txt": "hello\n"}, git=False)
    (tmp_path / "link.txt").symlink_to(tmp_path / "small.txt")
    result = scan_repo(tmp_path, use_git=False, max_file_bytes=100)
    assert sorted(result.files) == ["small.txt"]
    assert result.skipped["big.txt"].startswith("too large")
    assert result.skipped["link.txt"] == "symlink"


def test_scan_paths_inspects_only_the_given_paths(tmp_path: Path) -> None:
    init_repo(tmp_path, FILES, git=False)
    result = scan_paths(tmp_path, ["src/a.ts", "missing.ts", "logo.png"])
    assert sorted(result.files) == ["src/a.ts"]
    assert result.skipped["missing.ts"] == "missing or unreadable"


def test_scanned_file_carries_stat_data(tmp_path: Path) -> None:
    init_repo(tmp_path, {"a.txt": "hello\n"}, git=False)
    scanned = inspect_file(tmp_path, "a.txt", 1000)
    assert not isinstance(scanned, str)
    assert scanned.size == 6
    assert scanned.mtime_ns == os.stat(tmp_path / "a.txt").st_mtime_ns
