"""The storage seam.

Deliberately NOT a generic database abstraction: it names the operations the
engine needs, so another backend (Postgres + pgvector, an ANN store) can be
added later without touching retrieval. See docs/decisions/0002.

Contract highlights:
- `replace_file` is atomic across every table that derives from a file.
- Embeddings are supplied by the caller (embedding happens *before* the write
  transaction opens, never inside it).
- Search methods return ranked hits; scores are "higher is better".
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from kibble_engine.store.models import (
    CalleeRef,
    ChunkRecord,
    EdgeRecord,
    FileRecord,
    FileState,
    Hit,
    StoredChunk,
    SymbolRecord,
    SymbolRef,
)


class Store(Protocol):
    def replace_file(
        self,
        file: FileRecord,
        chunks: Sequence[ChunkRecord],
        symbols: Sequence[SymbolRecord] = (),
        edges: Sequence[EdgeRecord] = (),
    ) -> None:
        """Atomically replace everything derived from one file: manifest row,
        chunks, FTS rows, vectors, symbols, edges. All or nothing."""
        ...

    def remove_file(self, path: str) -> None: ...

    def manifest(self) -> dict[str, FileState]:
        """path -> state, for the query-time stat sweep."""
        ...

    def get_cached_embeddings(self, chunk_hashes: Sequence[str]) -> dict[str, list[float]]:
        """Embeddings already computed (for the current embedder) by chunk hash."""
        ...

    def vector_search(self, embedding: Sequence[float], k: int) -> list[Hit]: ...

    def keyword_search(self, query: str, k: int, *, match_all: bool = False) -> list[Hit]: ...

    def get_chunks(self, chunk_ids: Sequence[int]) -> list[StoredChunk]:
        """Chunks in the order requested; unknown ids are skipped."""
        ...

    def get_callers(self, symbol: str, limit: int = 50) -> list[SymbolRef]: ...

    def get_callees(self, symbol: str, limit: int = 50) -> list[CalleeRef]: ...

    def resolve_edges(self) -> int:
        """Link unresolved edges whose target name maps to exactly one symbol."""
        ...

    def get_meta(self, key: str) -> str | None: ...

    def set_meta(self, key: str, value: str) -> None: ...

    def counts(self) -> dict[str, int]: ...

    def close(self) -> None: ...
