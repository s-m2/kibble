"""The stat sweep: decide what changed since the last index, cheaply.

For every scanned file we compare (mtime, size) with the manifest. Equal means
unchanged and the file is never read. Different means we hash the content: identical
content is just a "touch" (a checkout or `touch` bumped the mtime), different content
is a real change.

The one hole in stat-based change detection is the *racy timestamp*: a file edited
again within the filesystem's timestamp granularity keeps the same mtime and (if the
size didn't change) looks untouched. Git has the same problem and the same fix: a
file whose mtime is not safely older than the moment we indexed it is hashed anyway.
Files edited long before they were indexed stay on the fast path.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from kibble_engine.core.scanner import ScannedFile, ScanResult
from kibble_engine.hashing import sha256_bytes_hex
from kibble_engine.store.models import FileState

RACY_WINDOW_S = 2


@dataclass
class SweepResult:
    new: list[ScannedFile] = field(default_factory=list)
    changed: list[ScannedFile] = field(default_factory=list)
    touched: list[ScannedFile] = field(default_factory=list)  # same content, new stat data
    deleted: list[str] = field(default_factory=list)
    unchanged: int = 0

    @property
    def dirty(self) -> list[ScannedFile]:
        """Files that need (re)indexing, in a stable order."""
        return sorted([*self.new, *self.changed], key=lambda f: f.path)


def _hash_file(path: Path) -> str | None:
    try:
        return sha256_bytes_hex(path.read_bytes())
    except OSError:
        return None


def sweep(
    root: Path,
    manifest: Mapping[str, FileState],
    scan: ScanResult,
    *,
    only: Collection[str] | None = None,
) -> SweepResult:
    """Compare a scan with the manifest. With `only`, restrict deletions to those paths
    (a targeted refresh must not conclude that every other file was deleted)."""
    result = SweepResult()
    for path, scanned in scan.files.items():
        state = manifest.get(path)
        if state is None:
            result.new.append(scanned)
            continue
        stat_same = state.mtime_ns == scanned.mtime_ns and state.size == scanned.size
        racy = scanned.mtime_ns >= (state.indexed_at - RACY_WINDOW_S) * 1_000_000_000
        if stat_same and not racy:
            result.unchanged += 1
            continue
        digest = _hash_file(root / path)
        if digest is None:
            continue  # vanished between scan and read; the next sweep will see it as deleted
        if digest != state.content_hash:
            result.changed.append(scanned)
        elif stat_same and not racy:
            result.unchanged += 1
        else:
            result.touched.append(scanned)  # re-stamp so the racy window moves forward

    candidates = manifest.keys() if only is None else [p for p in only if p in manifest]
    result.deleted = sorted(p for p in candidates if p not in scan.files)
    return result
