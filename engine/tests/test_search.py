"""Hybrid search end to end: real store and refresher, fake embedding server."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kibble_engine.core.refresh import Refresher
from kibble_engine.core.search import HybridSearcher, SearchConfig, SearchFilters
from kibble_engine.testing.fake_embedding_server import FakeEmbeddingServer
from kibble_engine.testing.helpers import init_repo, make_embedder, open_store, write

Env = tuple[FakeEmbeddingServer, HybridSearcher]

TOKEN_DOC = "def validate_token(token): check that the login token has not expired yet\n"
FILES = {
    "auth.txt": TOKEN_DOC,
    "billing.txt": "invoice total tax payment due customer ledger\n",
    "readme.txt": "project overview and installation notes for new contributors\n",
}


def make(tmp_path: Path, files: dict[str, str], *, perturb: float = 0.0, **cfg: Any) -> Env:
    init_repo(tmp_path, files, git=False)
    server = FakeEmbeddingServer(dim=128)
    server.perturb = perturb
    embedder = make_embedder(server)
    store = open_store(tmp_path, embedder)
    refresher = Refresher(tmp_path, store, embedder, use_git=False)
    refresher.refresh()
    return server, HybridSearcher(store, embedder, refresher, config=SearchConfig(**cfg))


def paths(response: Any) -> list[str]:
    return [h.chunk.file_path for h in response.hits]


def test_natural_language_query_finds_the_relevant_file_first(tmp_path: Path) -> None:
    _, searcher = make(tmp_path, FILES)
    response = searcher.search("how do we check that a token has not expired?")
    assert response.shape == "natural" and response.weights == (1.2, 0.8)
    assert paths(response)[0] == "auth.txt"
    top = response.hits[0]
    assert top.vector_rank is not None and top.keyword_rank is not None  # both signals agree


def test_exact_identifier_is_found_even_when_embeddings_are_useless(tmp_path: Path) -> None:
    files = {f"noise{i}.txt": f"generic filler number {i} about nothing\n" for i in range(8)}
    files["target.txt"] = "parseConfigV2 handles the legacy settings format\n"
    _, searcher = make(tmp_path, files, perturb=50.0)  # noise swamps the vector signal

    response = searcher.search("parseConfigV2")

    assert response.shape == "identifier" and response.weights == (0.6, 1.4)
    assert response.hits[0].chunk.file_path == "target.txt"
    assert response.hits[0].keyword_rank == 1


def test_single_signal_modes_use_only_that_signal(tmp_path: Path) -> None:
    _, searcher = make(tmp_path, FILES)
    vector_only = searcher.search("login token expired", config=SearchConfig(mode="vector"))
    assert vector_only.hits and all(h.keyword_rank is None for h in vector_only.hits)
    assert vector_only.keyword_query == "" and "keyword_search" not in vector_only.timings_ms

    keyword_only = searcher.search("login token expired", config=SearchConfig(mode="keyword"))
    assert keyword_only.hits and all(h.vector_rank is None for h in keyword_only.hits)
    assert "embed_query" not in keyword_only.timings_ms


def test_filters_apply_before_fusion(tmp_path: Path) -> None:
    files = {
        "src/a.txt": "shared searchable phrase appears here\n",
        "docs/a.txt": "shared searchable phrase appears here too\n",
        "tests/a.txt": "shared searchable phrase appears in tests\n",
    }
    _, searcher = make(tmp_path, files)
    only_docs = searcher.search(
        "shared searchable phrase", filters=SearchFilters(path_prefix="docs/")
    )
    assert paths(only_docs) == ["docs/a.txt"]

    no_tests = searcher.search(
        "shared searchable phrase", filters=SearchFilters(exclude_tags=frozenset({"test"}))
    )
    assert "tests/a.txt" not in paths(no_tests) and len(no_tests.hits) == 2


def test_test_and_vendor_files_are_down_weighted(tmp_path: Path) -> None:
    # Named so the penalised files sort (and tie-break) *before* the real source file:
    # without the penalty they would come first.
    same = "identical unique phrase present in every copy\n"
    _, searcher = make(
        tmp_path,
        {"e2e/core.txt": same, "site-packages/core.txt": same, "src/core.txt": same},
    )
    assert paths(searcher.search("identical unique phrase")) == [
        "src/core.txt",
        "e2e/core.txt",
        "site-packages/core.txt",
    ]


def test_asking_about_tests_removes_the_test_penalty(tmp_path: Path) -> None:
    _, searcher = make(tmp_path, {"tests/core.txt": "unique phrase here\n"})
    penalised = searcher.search("unique phrase").hits[0].score
    unpenalised = searcher.search("unique phrase tests").hits[0].score
    assert unpenalised / penalised == pytest.approx(1 / 0.8)


def test_one_file_cannot_fill_the_results(tmp_path: Path) -> None:
    paragraphs = [f"alphaterm number{i} " + "filler words go here " * 15 for i in range(6)]
    _, searcher = make(tmp_path, {"big.txt": "\n\n".join(paragraphs) + "\n"})
    capped = searcher.search("alphaterm", config=SearchConfig(max_per_file=2))
    assert len(capped.hits) == 2
    uncapped = searcher.search("alphaterm", config=SearchConfig(max_per_file=10))
    assert len(uncapped.hits) == 6


def test_embedder_outage_degrades_to_keyword_results(tmp_path: Path) -> None:
    server, searcher = make(tmp_path, FILES)
    server.fail_next(503, times=20)
    response = searcher.search("login token expired", fresh=False)
    assert response.degraded and "vector search unavailable" in response.degraded[0]
    assert paths(response)[0] == "auth.txt"
    assert all(h.vector_rank is None for h in response.hits)


def test_search_refreshes_first_and_reports_freshness(tmp_path: Path) -> None:
    _, searcher = make(tmp_path, FILES)
    write(tmp_path, "auth.txt", "completely rewritten uniqueneedleword content\n")

    stale = searcher.search("uniqueneedleword", fresh=False)
    assert stale.freshness is None  # the check was skipped...
    # ...so the edit is invisible (vector search still returns its nearest chunks anyway)
    assert not any("uniqueneedleword" in h.chunk.content for h in stale.hits)

    fresh = searcher.search("uniqueneedleword")
    top = fresh.hits[0]
    assert top.chunk.file_path == "auth.txt" and "uniqueneedleword" in top.chunk.content
    assert top.keyword_rank == 1
    assert fresh.freshness is not None
    assert fresh.freshness["refreshed_files"] == 1 and fresh.freshness["stale"] is False


def test_response_is_json_serialisable_with_timings(tmp_path: Path) -> None:
    _, searcher = make(tmp_path, FILES)
    response = searcher.search("login token")
    payload = json.loads(json.dumps(response.to_dict()))
    assert set(payload["timings_ms"]) >= {
        "refresh",
        "embed_query",
        "vector_search",
        "keyword_search",
        "fuse",
        "total",
    }
    hit = payload["hits"][0]
    assert {"path", "start_line", "end_line", "score", "vector_rank", "keyword_rank"} <= set(hit)
    assert payload["query_shape"] in {"identifier", "natural", "mixed"}


def test_empty_query_and_empty_index_are_handled(tmp_path: Path) -> None:
    _, searcher = make(tmp_path, FILES)
    empty = searcher.search("   ")
    assert empty.hits == [] and empty.degraded == ["empty query"]

    other = tmp_path / "other"
    _, blank = make(other, {})
    assert blank.search("anything at all").hits == []
