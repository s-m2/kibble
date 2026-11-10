"""The storage seam.

Deliberately NOT a generic database abstraction: it names the operations the
engine needs, so another backend (Postgres + pgvector, an ANN store) can be
added later without touching retrieval. See docs/decisions/0002.

Signatures are placeholders until the schema work lands; the method list is the
contract worth reviewing.
"""

from __future__ import annotations

from typing import Protocol


class Store(Protocol):
    def replace_file(self, path: str, content_hash: str) -> None:
        """Atomically replace everything derived from one file:
        manifest row, chunks, FTS rows, vectors, symbols, edges."""
        ...

    def vector_search(self, embedding: list[float], k: int) -> list[int]:
        """Return chunk ids ranked by vector similarity."""
        ...

    def keyword_search(self, query: str, k: int) -> list[int]:
        """Return chunk ids ranked by BM25."""
        ...

    def get_callers(self, symbol: str, limit: int) -> list[int]: ...

    def get_callees(self, symbol: str, limit: int) -> list[int]: ...

    def manifest(self) -> dict[str, str]:
        """path -> content hash, for the query-time stat sweep."""
        ...
