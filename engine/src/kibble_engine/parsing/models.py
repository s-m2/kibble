"""Outputs of the chunker. Embeddings are attached later (see core)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChunkSpec:
    """A chunk before embedding. `chunk_hash` is the hash of `embed_text`."""

    chunk_hash: str
    kind: str
    start_line: int
    end_line: int
    content: str
    embed_text: str
    symbol: str | None
    parent_symbol: str | None
    symbol_terms: str


@dataclass(frozen=True, slots=True)
class SymbolSpec:
    """A symbol defined in the file; `chunk_index` points into ParsedFile.chunks."""

    name: str
    kind: str
    start_line: int
    end_line: int
    chunk_index: int
    parent: str | None = None


@dataclass(frozen=True, slots=True)
class ParsedFile:
    language: str
    chunks: list[ChunkSpec]
    symbols: list[SymbolSpec]
    has_syntax_errors: bool = False
    used_fallback: bool = False
