"""Embedder interface and shared types."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol


def compute_embedder_id(
    model: str,
    dim: int,
    doc_prefix: str,
    dimensions: int | None,
    slice_dimensions: int | None = None,
) -> str:
    """Identify the *vectors* an embedder produces.

    Only settings that change stored vectors participate: model, output size, the
    document prefix, and the truncation settings. The base URL deliberately does
    not, so moving the same model from a hosted provider to a local server can reuse
    the embedding cache. The query prefix doesn't either: queries are never stored.
    `slice_dimensions` is only included when set, so ids made before it existed
    stay valid.
    """
    fields: dict[str, object] = {
        "model": model,
        "dim": dim,
        "doc_prefix": doc_prefix,
        "dimensions": dimensions,
    }
    if slice_dimensions is not None:
        fields["slice_dimensions"] = slice_dimensions
    payload = json.dumps(fields, sort_keys=True)
    return f"{model}@{dim}:{hashlib.sha256(payload.encode()).hexdigest()[:8]}"


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


@dataclass(frozen=True, slots=True)
class EmbedderInfo:
    embedder_id: str
    model: str
    dim: int
    host: str  # informational only (never part of the id)


@dataclass(frozen=True, slots=True)
class ProbeResult:
    dim: int
    latency_s: float


@dataclass
class EmbedderStats:
    """Counters the doctor command (and tests) use to see what the client had to do."""

    retries: int = 0
    batch_splits: int = 0
    input_shrinks: int = 0
    truncations: int = 0
    requests: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def bump(self, name: str, amount: int = 1) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + amount)


ProgressCallback = Callable[[int, int], None]  # (embedded so far, total)


class Embedder(Protocol):
    @property
    def info(self) -> EmbedderInfo: ...

    def probe(self) -> ProbeResult:
        """Make one real request; learn and validate the output dimension."""
        ...

    def embed_documents(
        self, texts: Sequence[str], *, on_progress: ProgressCallback | None = None
    ) -> list[list[float]]: ...

    def embed_queries(
        self, texts: Sequence[str], *, on_progress: ProgressCallback | None = None
    ) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...
