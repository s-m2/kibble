"""Keep the index true to the working tree.

`Refresher.refresh()` is the one code path behind every trigger: a full sweep at
startup, a query-time check, an idle timer, a git hook, an agent hook. Callers differ
only in *which paths* they pass (all of them, or the few they know changed) and *how many
files they can afford to index inline* (`max_files`). Whatever isn't done inline is
reported as `pending`, never silently skipped; the report says whether the index is
`stale` so answers can carry that fact to the agent.

Correctness never depends on an event source. Triggers only make the sweep cheaper by
telling it where to look; the sweep itself (stat comparison, see sweep.py) is what
guarantees a changed file is noticed.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kibble_engine.core.indexer import DEFAULT_WINDOW_CHUNKS, Indexer, ProgressCallback
from kibble_engine.core.scanner import (
    DEFAULT_MAX_FILE_BYTES,
    ScanResult,
    scan_paths,
    scan_repo,
)
from kibble_engine.core.sweep import SweepResult, sweep
from kibble_engine.embedders.base import Embedder
from kibble_engine.parsing import DEFAULT_MAX_CHARS, DEFAULT_MIN_CHARS
from kibble_engine.store.base import Store

DEFAULT_INLINE_FILES = 25  # query-time budget before work is deferred


@dataclass
class RefreshReport:
    scanned: int = 0
    unchanged: int = 0
    new: int = 0
    changed: int = 0
    touched: int = 0
    skipped_by_scan: int = 0
    refreshed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    aborted: str | None = None
    embedded_chunks: int = 0
    cached_chunks: int = 0
    duration_s: float = 0.0

    @property
    def stale(self) -> bool:
        """True if some file's index entry may not match the working tree."""
        return bool(self.pending or self.failed or self.aborted)

    def freshness(self) -> dict[str, Any]:
        """The metadata tool responses carry, so an agent can judge how far to trust them."""
        return {
            "stale": self.stale,
            "checked_files": self.scanned,
            "refreshed_files": len(self.refreshed),
            "removed_files": len(self.removed),
            "pending_files": len(self.pending),
            "failed_files": [{"path": p, "reason": r} for p, r in self.failed[:20]],
            "aborted": self.aborted,
            "duration_ms": round(self.duration_s * 1000),
        }


def _relative(root: Path, path: str | Path) -> str | None:
    """A repo-relative posix path, or None if `path` is outside the repository."""
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            candidate = candidate.relative_to(root)
        except ValueError:
            return None
    posix = candidate.as_posix()
    return None if posix.startswith("../") or posix == ".." else posix


def plan_changes(
    root: Path,
    store: Store,
    *,
    paths: Iterable[str | Path] | None = None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    use_git: bool = True,
) -> tuple[ScanResult, SweepResult]:
    """What a refresh would do, without doing it (no writes, no embedding)."""
    only: list[str] | None = None
    if paths is None:
        scan = scan_repo(root, max_file_bytes=max_file_bytes, use_git=use_git)
    else:
        only = [r for p in paths if (r := _relative(root, p)) is not None]
        scan = scan_paths(root, only, max_file_bytes=max_file_bytes)
    return scan, sweep(root, store.manifest(), scan, only=only)


class Refresher:
    def __init__(
        self,
        root: Path,
        store: Store,
        embedder: Embedder,
        *,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
        use_git: bool = True,
        max_chars: int = DEFAULT_MAX_CHARS,
        min_chars: int = DEFAULT_MIN_CHARS,
        window_chunks: int = DEFAULT_WINDOW_CHUNKS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._root = root
        self._store = store
        self._max_file_bytes = max_file_bytes
        self._use_git = use_git
        self._clock = clock
        self._indexer = Indexer(
            root,
            store,
            embedder,
            max_chars=max_chars,
            min_chars=min_chars,
            window_chunks=window_chunks,
        )
        self._lock = threading.RLock()  # one refresh at a time; concurrent callers queue

    def plan(self, paths: Iterable[str | Path] | None = None) -> tuple[ScanResult, SweepResult]:
        return plan_changes(
            self._root,
            self._store,
            paths=paths,
            max_file_bytes=self._max_file_bytes,
            use_git=self._use_git,
        )

    def refresh(
        self,
        paths: Iterable[str | Path] | None = None,
        *,
        max_files: int | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> RefreshReport:
        """Bring the index in line with the working tree.

        `paths=None` sweeps the whole repository; otherwise only those paths are
        examined (absent files among them count as deletions). `max_files` caps how many
        new/changed files are indexed in this call; the rest come back as `pending`.
        """
        started = self._clock()
        with self._lock:
            scan, result = self.plan(None if paths is None else list(paths))

            for path in result.deleted:
                self._store.remove_file(path)
            for touched in result.touched:
                self._store.touch_file(touched.path, touched.mtime_ns, touched.size)

            dirty = result.dirty
            inline = dirty if max_files is None else dirty[:max_files]
            deferred = [] if max_files is None else dirty[max_files:]
            run = self._indexer.index_files(inline, on_progress=on_progress)

        return RefreshReport(
            scanned=len(scan.files),
            unchanged=result.unchanged,
            new=len(result.new),
            changed=len(result.changed),
            touched=len(result.touched),
            skipped_by_scan=len(scan.skipped),
            refreshed=[o.path for o in run.outcomes],
            removed=list(result.deleted),
            failed=list(run.failed),
            pending=[*run.pending, *(f.path for f in deferred)],
            aborted=run.aborted,
            embedded_chunks=run.embedded_chunks,
            cached_chunks=run.cached_chunks,
            duration_s=self._clock() - started,
        )

    def ensure_fresh(
        self,
        paths: Iterable[str | Path] | None = None,
        *,
        max_files: int | None = DEFAULT_INLINE_FILES,
    ) -> RefreshReport:
        """The query-time check: refresh now, within a budget, and say what is left over."""
        return self.refresh(paths, max_files=max_files)
