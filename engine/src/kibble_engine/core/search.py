"""Hybrid search: vector similarity and BM25, fused by reciprocal rank.

Why two signals: embeddings find *intent* ("where do we handle expired logins?") but are
weak on rare exact identifiers; BM25 is the opposite. Their scores are incomparable
(cosine vs BM25), so they are merged by *rank* with weighted reciprocal rank fusion:

    score(chunk) = sum over lists of  weight / (k + rank_in_list)

A chunk both lists like beats one only a single list likes, which filters out the
"plausible but wrong" vector hits that have no lexical support.

Around the fusion:

- **Query shape** (identifier-like, natural language, mixed) nudges the list weights.
- **Keyword queries are prepared**: stopwords dropped from natural-language questions,
  `camelCase` / `snake_case` tokens also searched as their parts.
- **Filters** apply to each list *before* fusion, so ranks are within what the caller wants.
- **Tags** (`test`, `vendor`, `generated`) down-weight noise unless the query asks for it.
- **Diversity**: at most N hits per file, and overlapping chunks of one file collapse.
- **Degradation, not failure**: if the embedder is unreachable, keyword results are
  returned with a note saying so.
- **Freshness**: unless told otherwise, the index is refreshed first and the response says
  how fresh it is.

Every default weight and threshold here is a starting guess. Each one has a switch so the
eval harness can ablate it; none is a result.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from kibble_engine.core.refresh import DEFAULT_INLINE_FILES, Refresher
from kibble_engine.embedders.base import Embedder
from kibble_engine.embedders.errors import EmbedderError
from kibble_engine.parsing.identifiers import split_identifier
from kibble_engine.store.base import Store
from kibble_engine.store.models import Hit, StoredChunk

DEFAULT_RRF_K = 60  # the constant from the original RRF paper; larger flattens the tail
DEFAULT_CANDIDATES = 50

STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from how i if in is it its of on or "
    "that the this to was what when where which who why with without you your".split()
)
_TEST_WORDS = frozenset({"test", "tests", "spec", "specs", "pytest", "vitest", "jest", "mock"})
_EDGE_PUNCTUATION = "?!,;:()\"'`"

QueryKind = Literal["identifier", "natural", "mixed"]
Mode = Literal["hybrid", "vector", "keyword"]


# --------------------------------------------------------------- query analysis


@dataclass(frozen=True, slots=True)
class QueryShape:
    kind: QueryKind
    tokens: tuple[str, ...]


def _is_code_like(token: str) -> bool:
    """`parseConfig`, `snake_case`, `a.b`, `Foo::bar`, `x()`, `v2`: looks like an identifier."""
    return bool(
        re.search(r"[a-z][A-Z]", token)
        or any(c in token for c in "_.()")
        or "::" in token
        or "->" in token
        or (re.search(r"\d", token) and re.search(r"[A-Za-z]", token))
    )


def analyze_query(query: str) -> QueryShape:
    tokens = tuple(t.strip(_EDGE_PUNCTUATION) for t in query.split() if t.strip(_EDGE_PUNCTUATION))
    quoted = '"' in query or "`" in query
    code_like = [t for t in tokens if _is_code_like(t)]
    stop_fraction = sum(t.lower() in STOPWORDS for t in tokens) / len(tokens) if tokens else 0.0

    if len(tokens) <= 2 and (code_like or quoted):
        return QueryShape("identifier", tokens)
    if len(tokens) >= 4 and (stop_fraction >= 0.25 or query.rstrip().endswith("?")):
        return QueryShape("natural", tokens)
    return QueryShape("mixed", tokens)


def keyword_query(
    query: str, shape: QueryShape, *, drop_stopwords: bool = True, expand_identifiers: bool = True
) -> str:
    """The text handed to BM25. The store quotes each term and ORs them."""
    terms = list(shape.tokens)
    if drop_stopwords and shape.kind != "identifier" and len(terms) >= 3:
        kept = [t for t in terms if t.lower() not in STOPWORDS]
        terms = kept or terms  # a query made only of stopwords is better than no query
    if expand_identifiers:
        seen = {t.lower() for t in terms}
        for token in list(terms):
            if not _is_code_like(token):
                continue
            parts = split_identifier(token)
            if len(parts) > 1:
                for part in parts:
                    if part not in seen:
                        seen.add(part)
                        terms.append(part)
    return " ".join(terms) if terms else query


# ------------------------------------------------------------------- fusion


def rrf_fuse(
    rankings: Sequence[Sequence[int]],
    weights: Sequence[float],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[tuple[int, float]]:
    """Weighted reciprocal rank fusion. Returns (id, score), best first.

    Ties are broken by the best rank any list gave the item, then by id, so results are
    deterministic.
    """
    scores: dict[int, float] = {}
    best_rank: dict[int, int] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        seen: set[int] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen:
                continue
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
            best_rank[item] = min(best_rank.get(item, rank), rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], best_rank[kv[0]], kv[0]))


# ------------------------------------------------------------------- results


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk: StoredChunk
    score: float
    vector_rank: int | None = None  # 1-based position in the (filtered) vector list
    keyword_rank: int | None = None
    vector_score: float | None = None  # cosine similarity
    keyword_score: float | None = None  # -bm25

    def to_dict(self) -> dict[str, Any]:
        c = self.chunk
        return {
            "chunk_id": c.id,
            "path": c.file_path,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "kind": c.kind,
            "symbol": c.symbol,
            "parent_symbol": c.parent_symbol,
            "language": c.language,
            "tags": c.tags.split(),
            "content": c.content,
            "score": self.score,
            "vector_rank": self.vector_rank,
            "keyword_rank": self.keyword_rank,
        }


@dataclass
class SearchResponse:
    query: str
    hits: list[SearchHit] = field(default_factory=list)
    freshness: dict[str, Any] | None = None
    degraded: list[str] = field(default_factory=list)  # reasons results are partial
    shape: QueryKind = "mixed"
    weights: tuple[float, float] = (1.0, 1.0)  # (vector, keyword) actually used
    keyword_query: str = ""
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "hits": [h.to_dict() for h in self.hits],
            "freshness": self.freshness,
            "degraded": self.degraded,
            "query_shape": self.shape,
            "weights": {"vector": self.weights[0], "keyword": self.weights[1]},
            "keyword_query": self.keyword_query,
            "timings_ms": self.timings_ms,
        }


# ------------------------------------------------------------------- config


@dataclass(frozen=True, slots=True)
class SearchFilters:
    path_prefix: str | None = None
    languages: frozenset[str] | None = None
    exclude_tags: frozenset[str] = frozenset()

    @property
    def active(self) -> bool:
        return bool(self.path_prefix or self.languages or self.exclude_tags)

    def allows(self, chunk: StoredChunk) -> bool:
        if self.path_prefix and not chunk.file_path.startswith(self.path_prefix):
            return False
        if self.languages is not None and chunk.language not in self.languages:
            return False
        return not (self.exclude_tags & set(chunk.tags.split()))


def _default_shape_weights() -> dict[str, tuple[float, float]]:
    # (vector multiplier, keyword multiplier). Untuned starting guesses.
    return {"identifier": (0.6, 1.4), "natural": (1.2, 0.8), "mixed": (1.0, 1.0)}


def _default_tag_weights() -> dict[str, float]:
    return {"vendor": 0.5, "generated": 0.5, "test": 0.8}


@dataclass(frozen=True)
class SearchConfig:
    k: int = 10
    candidates: int = DEFAULT_CANDIDATES
    rrf_k: int = DEFAULT_RRF_K
    mode: Mode = "hybrid"
    vector_weight: float = 1.0
    keyword_weight: float = 1.0
    shape_weighting: bool = True
    shape_weights: Mapping[str, tuple[float, float]] = field(default_factory=_default_shape_weights)
    drop_stopwords: bool = True
    expand_identifiers: bool = True
    tag_weights: Mapping[str, float] = field(default_factory=_default_tag_weights)
    max_per_file: int = 3
    collapse_overlaps: bool = True


# ----------------------------------------------------------------- searcher


class HybridSearcher:
    def __init__(
        self,
        store: Store,
        embedder: Embedder,
        refresher: Refresher | None = None,
        *,
        config: SearchConfig | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._refresher = refresher
        self._config = config or SearchConfig()
        self._clock = clock

    def search(
        self,
        query: str,
        *,
        config: SearchConfig | None = None,
        filters: SearchFilters | None = None,
        fresh: bool = True,
        max_refresh_files: int | None = DEFAULT_INLINE_FILES,
    ) -> SearchResponse:
        cfg = config or self._config
        filters = filters or SearchFilters()
        response = SearchResponse(query=query)
        timings = response.timings_ms
        started = self._clock()
        if not query.strip():
            response.degraded.append("empty query")
            return response

        if fresh and self._refresher is not None:
            t = self._clock()
            report = self._refresher.ensure_fresh(max_files=max_refresh_files)
            response.freshness = report.freshness()
            timings["refresh"] = (self._clock() - t) * 1000

        shape = analyze_query(query)
        response.shape = shape.kind
        wanted = cfg.candidates * (3 if filters.active else 1)

        vector_hits: list[Hit] = []
        if cfg.mode in ("hybrid", "vector"):
            t = self._clock()
            try:
                vector = self._embedder.embed_query(query)
                timings["embed_query"] = (self._clock() - t) * 1000
                t = self._clock()
                vector_hits = self._store.vector_search(vector, wanted)
                timings["vector_search"] = (self._clock() - t) * 1000
            except EmbedderError as exc:
                response.degraded.append(f"vector search unavailable: {exc}")

        keyword_hits: list[Hit] = []
        if cfg.mode in ("hybrid", "keyword"):
            t = self._clock()
            response.keyword_query = keyword_query(
                query,
                shape,
                drop_stopwords=cfg.drop_stopwords,
                expand_identifiers=cfg.expand_identifiers,
            )
            keyword_hits = self._store.keyword_search(response.keyword_query, wanted)
            timings["keyword_search"] = (self._clock() - t) * 1000

        t = self._clock()
        response.hits, response.weights = self._fuse(
            query, shape, vector_hits, keyword_hits, filters, cfg
        )
        timings["fuse"] = (self._clock() - t) * 1000
        timings["total"] = (self._clock() - started) * 1000
        return response

    # ------------------------------------------------------------------ fusion

    def _fuse(
        self,
        query: str,
        shape: QueryShape,
        vector_hits: list[Hit],
        keyword_hits: list[Hit],
        filters: SearchFilters,
        cfg: SearchConfig,
    ) -> tuple[list[SearchHit], tuple[float, float]]:
        ids = list(dict.fromkeys([h.chunk_id for h in vector_hits + keyword_hits]))
        chunks = {c.id: c for c in self._store.get_chunks(ids)}

        def visible(hits: list[Hit]) -> list[Hit]:
            kept = [h for h in hits if h.chunk_id in chunks and filters.allows(chunks[h.chunk_id])]
            return kept[: cfg.candidates]

        vector_list, keyword_list = visible(vector_hits), visible(keyword_hits)
        v_weight, k_weight = cfg.vector_weight, cfg.keyword_weight
        if cfg.shape_weighting:
            v_mult, k_mult = cfg.shape_weights.get(shape.kind, (1.0, 1.0))
            v_weight, k_weight = v_weight * v_mult, k_weight * k_mult

        fused = rrf_fuse(
            [[h.chunk_id for h in vector_list], [h.chunk_id for h in keyword_list]],
            [v_weight, k_weight],
            k=cfg.rrf_k,
        )
        v_info = {h.chunk_id: (rank, h.score) for rank, h in enumerate(vector_list, start=1)}
        k_info = {h.chunk_id: (rank, h.score) for rank, h in enumerate(keyword_list, start=1)}

        wants_tests = bool(_TEST_WORDS & {t.lower() for t in shape.tokens})
        rescored: list[tuple[int, float]] = []
        for chunk_id, score in fused:
            for tag in chunks[chunk_id].tags.split():
                if tag == "test" and wants_tests:
                    continue  # the query asks about tests: don't down-weight them
                score *= cfg.tag_weights.get(tag, 1.0)
            rescored.append((chunk_id, score))
        rescored.sort(key=lambda item: -item[1])  # stable: fusion tie-breaks survive

        chosen = _diversify(rescored, chunks, cfg)
        hits = [
            SearchHit(
                chunks[chunk_id],
                score,
                v_info.get(chunk_id, (None, None))[0],
                k_info.get(chunk_id, (None, None))[0],
                v_info.get(chunk_id, (None, None))[1],
                k_info.get(chunk_id, (None, None))[1],
            )
            for chunk_id, score in chosen
        ]
        return hits, (v_weight, k_weight)


def _overlap(a: StoredChunk, b: StoredChunk) -> float:
    """Fraction of the shorter chunk's lines that the other covers."""
    low, high = max(a.start_line, b.start_line), min(a.end_line, b.end_line)
    if high < low:
        return 0.0
    shorter = min(a.end_line - a.start_line, b.end_line - b.start_line) + 1
    return (high - low + 1) / shorter


def _diversify(
    ranked: list[tuple[int, float]], chunks: Mapping[int, StoredChunk], cfg: SearchConfig
) -> list[tuple[int, float]]:
    """Cap hits per file and collapse chunks of one file that overlap heavily."""
    chosen: list[tuple[int, float]] = []
    per_file: dict[str, list[StoredChunk]] = {}
    for chunk_id, score in ranked:
        chunk = chunks[chunk_id]
        taken = per_file.setdefault(chunk.file_path, [])
        if len(taken) >= cfg.max_per_file:
            continue
        if cfg.collapse_overlaps and any(_overlap(chunk, other) >= 0.5 for other in taken):
            continue
        taken.append(chunk)
        chosen.append((chunk_id, score))
        if len(chosen) >= cfg.k:
            break
    return chosen
