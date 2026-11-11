"""Plain data types exchanged with a Store. No behaviour, no database types."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FileRecord:
    """A file to (re)index. `path` is repo-relative with forward slashes."""

    path: str
    content_hash: str
    mtime_ns: int
    size: int
    language: str | None = None
    tags: str = ""


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """One chunk plus its embedding, ready to store.

    `chunk_hash` is the hash of `embed_text` (see kibble_engine.hashing). The
    embedding need not be normalised; the store normalises it.
    """

    chunk_hash: str
    kind: str
    start_line: int
    end_line: int
    content: str
    embed_text: str
    embedding: Sequence[float]
    symbol: str | None = None
    parent_symbol: str | None = None
    symbol_terms: str = ""


@dataclass(frozen=True, slots=True)
class SymbolRecord:
    """A symbol defined in the file. `chunk_index` indexes the chunks passed alongside."""

    name: str
    kind: str
    start_line: int
    end_line: int
    chunk_index: int | None = None


@dataclass(frozen=True, slots=True)
class EdgeRecord:
    """An outgoing edge. `source_index` indexes the symbols passed alongside."""

    source_index: int
    kind: str
    target_name: str


@dataclass(frozen=True, slots=True)
class FileState:
    """What the manifest remembers about a file (for the stat sweep)."""

    content_hash: str
    mtime_ns: int
    size: int


@dataclass(frozen=True, slots=True)
class Hit:
    """A search hit. `score`: higher is better (cosine similarity, or -bm25)."""

    chunk_id: int
    score: float


@dataclass(frozen=True, slots=True)
class StoredChunk:
    id: int
    file_path: str
    kind: str
    symbol: str | None
    parent_symbol: str | None
    start_line: int
    end_line: int
    content: str


@dataclass(frozen=True, slots=True)
class SymbolRef:
    id: int
    name: str
    kind: str
    file_path: str
    start_line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class CalleeRef:
    """A call target: the name as written, and the symbol it resolved to (if unique)."""

    target_name: str
    kind: str
    target: SymbolRef | None
