"""Turn files into index entries: read, chunk, embed, `replace_file`.

Two properties matter more than speed:

**Embeddings are batched across files.** Quota-limited endpoints count *requests*, so
embedding one file at a time (a few chunks per call) would burn a free tier's daily
allowance on a few dozen files. Files are processed in *windows*: chunk several files,
look up cached vectors, embed every still-missing chunk of the window in one call
(the embedder splits that into big batches), then commit each file separately.

**Commits are per file and failures are conservative.** `replace_file` is atomic, so a
file is either fully re-indexed or left exactly as it was. If the embedder fails, files
in flight keep their old index entries and are reported as stale; they are never
half-written. Errors that will keep happening (auth, credits, rate limits, outages) stop
the run and mark everything not yet done as pending. A request the server rejects is
isolated to the file that caused it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from kibble_engine.core.scanner import ScannedFile
from kibble_engine.embedders.base import Embedder
from kibble_engine.embedders.errors import EmbedderError, EmbedderRequestError
from kibble_engine.hashing import sha256_bytes_hex
from kibble_engine.parsing import DEFAULT_MAX_CHARS, DEFAULT_MIN_CHARS, chunk_file
from kibble_engine.parsing.models import ParsedFile
from kibble_engine.store.base import Store
from kibble_engine.store.models import ChunkRecord, FileRecord, SymbolRecord

log = logging.getLogger(__name__)

DEFAULT_WINDOW_CHUNKS = 256
DEFAULT_WINDOW_FILES = 64
_BINARY_SNIFF_BYTES = 8192

ProgressCallback = Callable[[int, int], None]  # (files finished, files total)


@dataclass(frozen=True, slots=True)
class IndexOutcome:
    path: str
    status: Literal["indexed", "empty", "binary"]
    chunks: int = 0
    embedded: int = 0  # chunks that needed a new embedding
    cached: int = 0  # chunks served from the embedding cache


@dataclass
class IndexRun:
    outcomes: list[IndexOutcome] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (path, reason)
    pending: list[str] = field(default_factory=list)  # not attempted, or retryable
    aborted: str | None = None  # why the run stopped early, if it did

    @property
    def embedded_chunks(self) -> int:
        return sum(o.embedded for o in self.outcomes)

    @property
    def cached_chunks(self) -> int:
        return sum(o.cached for o in self.outcomes)


@dataclass
class _Prepared:
    scanned: ScannedFile
    record: FileRecord
    parsed: ParsedFile | None  # None for binary files (recorded without chunks)
    status: Literal["indexed", "empty", "binary"]
    cached: dict[str, list[float]] = field(default_factory=dict)
    missing: dict[str, str] = field(default_factory=dict)  # chunk_hash -> embed_text


class Indexer:
    def __init__(
        self,
        root: Path,
        store: Store,
        embedder: Embedder,
        *,
        max_chars: int = DEFAULT_MAX_CHARS,
        min_chars: int = DEFAULT_MIN_CHARS,
        window_chunks: int = DEFAULT_WINDOW_CHUNKS,
        window_files: int = DEFAULT_WINDOW_FILES,
    ) -> None:
        self._root = root
        self._store = store
        self._embedder = embedder
        self._max_chars = max_chars
        self._min_chars = min_chars
        self._window_chunks = max(1, window_chunks)
        self._window_files = max(1, window_files)

    # ----------------------------------------------------------------- public

    def index_file(self, scanned: ScannedFile) -> IndexRun:
        return self.index_files([scanned])

    def index_files(
        self, files: Sequence[ScannedFile], *, on_progress: ProgressCallback | None = None
    ) -> IndexRun:
        run = IndexRun()
        window: list[_Prepared] = []
        window_missing: dict[str, str] = {}
        total = len(files)

        def report() -> None:
            if on_progress is not None:
                finished = len(run.outcomes) + len(run.failed)
                on_progress(finished, total)

        for position, scanned in enumerate(files):
            try:
                prepared = self._prepare(scanned)
            except OSError as exc:
                run.failed.append((scanned.path, f"cannot read: {exc}"))
                report()
                continue
            window.append(prepared)
            window_missing.update(prepared.missing)

            if len(window_missing) >= self._window_chunks or len(window) >= self._window_files:
                if not self._flush(window, run):
                    run.pending.extend(f.path for f in files[position + 1 :])
                    report()
                    return run
                window, window_missing = [], {}
                report()

        if window and not self._flush(window, run):
            report()
            return run
        report()
        return run

    # ---------------------------------------------------------------- internals

    def _prepare(self, scanned: ScannedFile) -> _Prepared:
        path = self._root / scanned.path
        stat = path.stat()  # BEFORE reading: a concurrent edit then shows up as a newer mtime
        data = path.read_bytes()
        record = FileRecord(
            path=scanned.path,
            content_hash=sha256_bytes_hex(data),
            mtime_ns=stat.st_mtime_ns,
            size=stat.st_size,
            tags=scanned.tags,
        )
        if b"\0" in data[:_BINARY_SNIFF_BYTES]:
            # Remember it (hash + stat) with no chunks, so it isn't re-read on every sweep.
            return _Prepared(scanned, record, None, "binary")

        parsed = chunk_file(
            scanned.path,
            data.decode("utf-8", errors="replace"),
            max_chars=self._max_chars,
            min_chars=self._min_chars,
        )
        record = FileRecord(
            path=record.path,
            content_hash=record.content_hash,
            mtime_ns=record.mtime_ns,
            size=record.size,
            language=parsed.language,
            tags=record.tags,
        )
        if not parsed.chunks:
            return _Prepared(scanned, record, parsed, "empty")

        hashes = [c.chunk_hash for c in parsed.chunks]
        cached = self._store.get_cached_embeddings(hashes)
        missing = {c.chunk_hash: c.embed_text for c in parsed.chunks if c.chunk_hash not in cached}
        return _Prepared(scanned, record, parsed, "indexed", cached, missing)

    def _flush(self, window: list[_Prepared], run: IndexRun) -> bool:
        """Embed and commit one window. Returns False if the run must stop."""
        vectors = self._embed_window(window, run)
        if vectors is None:
            return False
        failed_paths = {path for path, _ in run.failed}
        for prepared in window:
            if prepared.scanned.path not in failed_paths:
                self._commit(prepared, vectors, run)
        return True

    def _embed_window(
        self, window: list[_Prepared], run: IndexRun
    ) -> dict[str, list[float]] | None:
        wanted: dict[str, str] = {}
        for prepared in window:
            wanted.update(prepared.missing)
        if not wanted:
            return {}
        hashes = list(wanted)
        try:
            embedded = self._embedder.embed_documents([wanted[h] for h in hashes])
            return dict(zip(hashes, embedded, strict=True))
        except EmbedderRequestError:
            return self._embed_isolating(window, run)
        except EmbedderError as exc:
            # Auth, credits, rate limits, outages, bad responses: it will keep failing,
            # so stop. Nothing in this window was written; it all stays as it was.
            run.aborted = str(exc)
            run.pending.extend(p.scanned.path for p in window)
            log.warning("indexing stopped: %s", exc)
            return None

    def _embed_isolating(
        self, window: list[_Prepared], run: IndexRun
    ) -> dict[str, list[float]] | None:
        """The server rejected something in the window: find which file(s) caused it."""
        vectors: dict[str, list[float]] = {}
        for prepared in window:
            if not prepared.missing:
                continue
            hashes = list(prepared.missing)
            try:
                embedded = self._embedder.embed_documents([prepared.missing[h] for h in hashes])
            except EmbedderRequestError as exc:
                run.failed.append((prepared.scanned.path, f"embedding rejected: {exc}"))
            except EmbedderError as exc:
                run.aborted = str(exc)
                run.pending.extend(p.scanned.path for p in window)
                return None
            else:
                vectors.update(zip(hashes, embedded, strict=True))
        return vectors

    def _commit(self, prepared: _Prepared, vectors: dict[str, list[float]], run: IndexRun) -> None:
        chunks: list[ChunkRecord] = []
        symbols: list[SymbolRecord] = []
        embedded = cached = 0
        if prepared.parsed is not None:
            for spec in prepared.parsed.chunks:
                vector = prepared.cached.get(spec.chunk_hash)
                if vector is None:
                    vector = vectors[spec.chunk_hash]
                    embedded += 1
                else:
                    cached += 1
                chunks.append(
                    ChunkRecord(
                        chunk_hash=spec.chunk_hash,
                        kind=spec.kind,
                        start_line=spec.start_line,
                        end_line=spec.end_line,
                        content=spec.content,
                        embed_text=spec.embed_text,
                        embedding=vector,
                        symbol=spec.symbol,
                        parent_symbol=spec.parent_symbol,
                        symbol_terms=spec.symbol_terms,
                    )
                )
            symbols = [
                SymbolRecord(s.name, s.kind, s.start_line, s.end_line, s.chunk_index)
                for s in prepared.parsed.symbols
            ]
        try:
            self._store.replace_file(prepared.record, chunks, symbols)
        except Exception as exc:  # one bad file must not sink the run
            log.exception("could not store %s", prepared.scanned.path)
            run.failed.append((prepared.scanned.path, f"store error: {exc}"))
            return
        run.outcomes.append(
            IndexOutcome(prepared.scanned.path, prepared.status, len(chunks), embedded, cached)
        )
