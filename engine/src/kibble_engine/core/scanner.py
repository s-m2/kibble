"""Which files belong in the index.

Inside a git repository the answer is git's: tracked files plus untracked files that
aren't ignored (`git ls-files --cached --others --exclude-standard`), so `.gitignore`
is respected without re-implementing it. Elsewhere we walk the tree, pruning the usual
junk directories. Either way, cheap stat-level filters then drop files that should
never be embedded: binaries by extension, lockfiles and minified bundles, symlinks,
empty files, and anything over the size cap.

The scanner never reads file contents; the indexer detects binary content later.
"""

from __future__ import annotations

import fnmatch
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

DEFAULT_MAX_FILE_BYTES = 1_000_000

PRUNED_DIRS = frozenset(
    {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        "dist",
        "build",
        ".kibble",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "target",
        "coverage",
        ".next",
    }
)

BINARY_EXTENSIONS = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".ico",
        ".webp",
        ".tiff",
        ".psd",
        ".mp3",
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".wav",
        ".flac",
        ".ogg",
        ".webm",
        ".zip",
        ".gz",
        ".tgz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        ".tar",
        ".jar",
        ".war",
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
        ".eot",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".a",
        ".o",
        ".obj",
        ".class",
        ".pyc",
        ".pyo",
        ".wasm",
        ".db",
        ".sqlite",
        ".sqlite3",
        ".bin",
        ".dat",
        ".pkl",
        ".npy",
        ".onnx",
        ".pt",
        ".safetensors",
    }
)

LOCKFILES = frozenset(
    {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "npm-shrinkwrap.json",
        "uv.lock",
        "poetry.lock",
        "Pipfile.lock",
        "Cargo.lock",
        "Gemfile.lock",
        "composer.lock",
        "go.sum",
    }
)
GENERATED_GLOBS = ("*.min.js", "*.min.css", "*.map", "*.bundle.js", "*.chunk.js")

_VENDOR_DIRS = frozenset({"node_modules", "vendor", "third_party", "third-party", "site-packages"})
_TEST_DIRS = frozenset({"test", "tests", "__tests__", "spec", "specs", "e2e"})
_TEST_GLOBS = ("test_*.py", "*_test.py", "*.test.*", "*.spec.*", "*_spec.*", "conftest.py")
_GENERATED_DIRS = frozenset({"generated", "__generated__"})


@dataclass(frozen=True, slots=True)
class ScannedFile:
    path: str  # repo-relative, forward slashes
    size: int
    mtime_ns: int
    tags: str = ""


@dataclass
class ScanResult:
    files: dict[str, ScannedFile] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)  # path -> reason
    used_git: bool = False


def classify_tags(path: str) -> str:
    """Cheap path-based labels ('test', 'vendor', 'generated') for later down-weighting."""
    posix = PurePosixPath(path)
    parts = {p.lower() for p in posix.parts[:-1]}
    name = posix.name
    tags = set()
    if parts & _VENDOR_DIRS:
        tags.add("vendor")
    if parts & _TEST_DIRS or any(fnmatch.fnmatch(name, g) for g in _TEST_GLOBS):
        tags.add("test")
    if parts & _GENERATED_DIRS or ".generated." in name or name.endswith("_pb2.py"):
        tags.add("generated")
    return " ".join(sorted(tags))


def inspect_file(root: Path, rel: str, max_file_bytes: int) -> ScannedFile | str:
    """Stat one candidate. Returns a ScannedFile, or a string explaining why it is skipped."""
    posix = PurePosixPath(rel)
    name = posix.name
    if posix.suffix.lower() in BINARY_EXTENSIONS:
        return "binary extension"
    if name in LOCKFILES or any(fnmatch.fnmatch(name, g) for g in GENERATED_GLOBS):
        return "lockfile or generated bundle"
    full = root / rel
    try:
        if full.is_symlink():
            return "symlink"
        st = full.stat()
    except OSError:
        return "missing or unreadable"
    if not full.is_file():
        return "not a regular file"
    if st.st_size == 0:
        return "empty"
    if st.st_size > max_file_bytes:
        return f"too large ({st.st_size} bytes)"
    return ScannedFile(rel, st.st_size, st.st_mtime_ns, classify_tags(rel))


def scan_repo(
    root: Path,
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    use_git: bool = True,
) -> ScanResult:
    """Enumerate indexable files under `root`."""
    result = ScanResult()
    candidates = _git_candidates(root) if use_git else None
    if candidates is not None:
        result.used_git = True
    else:
        candidates = _walk_candidates(root)
    return _filter(root, candidates, max_file_bytes, result)


def scan_paths(
    root: Path, paths: list[str], *, max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
) -> ScanResult:
    """Inspect just `paths` (repo-relative). Used by targeted refreshes."""
    return _filter(root, paths, max_file_bytes, ScanResult())


def _filter(
    root: Path, candidates: list[str], max_file_bytes: int, result: ScanResult
) -> ScanResult:
    for rel in sorted(set(candidates)):
        if rel == ".kibble" or rel.startswith(".kibble/"):
            continue  # never index our own index
        outcome = inspect_file(root, rel, max_file_bytes)
        if isinstance(outcome, ScannedFile):
            result.files[rel] = outcome
        else:
            result.skipped[rel] = outcome
    return result


def _git_candidates(root: Path) -> list[str] | None:
    """Files per git, or None when `root` isn't in a repository (or git is unavailable)."""
    try:
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return [p for p in proc.stdout.decode("utf-8", errors="replace").split("\0") if p]


def _walk_candidates(root: Path) -> list[str]:
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in PRUNED_DIRS]
        base = Path(dirpath).relative_to(root)
        found.extend((base / name).as_posix() for name in filenames)
    return found
