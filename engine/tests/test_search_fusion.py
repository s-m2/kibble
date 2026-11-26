"""The pure parts of hybrid search: fusion math, query analysis, keyword queries, diversity."""

from __future__ import annotations

import pytest

from kibble_engine.core.search import (
    SearchConfig,
    SearchFilters,
    _diversify,
    _overlap,
    analyze_query,
    keyword_query,
    rrf_fuse,
)
from kibble_engine.store.models import StoredChunk


def chunk(
    chunk_id: int,
    path: str = "a.ts",
    start: int = 1,
    end: int = 10,
    tags: str = "",
    lang: str = "ts",
) -> StoredChunk:
    return StoredChunk(chunk_id, path, "function", None, None, start, end, "body", tags, lang)


# ---------------------------------------------------------------------- RRF


def test_a_single_list_keeps_its_order_and_uses_the_rrf_formula() -> None:
    fused = rrf_fuse([[7, 3, 9]], [1.0], k=60)
    assert [i for i, _ in fused] == [7, 3, 9]
    assert fused[0][1] == pytest.approx(1 / 61)
    assert fused[2][1] == pytest.approx(1 / 63)


def test_agreement_between_lists_beats_a_single_list_win() -> None:
    # 1 tops list one, 3 tops list two, but 2 is second in both.
    fused = rrf_fuse([[1, 2], [3, 2]], [1.0, 1.0])
    assert fused[0][0] == 2
    assert fused[0][1] == pytest.approx(2 / 62)


def test_weights_shift_the_balance_between_lists() -> None:
    fused = rrf_fuse([[1], [2]], [1.0, 3.0])
    assert [i for i, _ in fused] == [2, 1]
    assert fused[0][1] == pytest.approx(3 / 61)


def test_ties_are_broken_deterministically() -> None:
    fused = rrf_fuse([[5], [3]], [1.0, 1.0])  # equal scores, equal best rank: lower id first
    assert [i for i, _ in fused] == [3, 5]
    assert fused == rrf_fuse([[5], [3]], [1.0, 1.0])


def test_larger_k_flattens_the_gap_between_ranks() -> None:
    for k in (1, 60, 1000):
        (_, first), (_, second) = rrf_fuse([[1, 2]], [1.0], k=k)
        assert first / second == pytest.approx((k + 2) / (k + 1))


def test_duplicates_within_a_list_count_once_and_empty_input_is_fine() -> None:
    assert rrf_fuse([[1, 1, 1]], [1.0]) == [(1, pytest.approx(1 / 61))]
    assert rrf_fuse([[], []], [1.0, 1.0]) == []


# ------------------------------------------------------------ query analysis


@pytest.mark.parametrize(
    ("query", "kind"),
    [
        ("validateToken", "identifier"),
        ("auth.validate", "identifier"),
        ('"user_id"', "identifier"),
        ("parse_config()", "identifier"),
        ("where do we handle expired logins?", "natural"),
        ("how does the session refresh work", "natural"),
        ("validate token", "mixed"),
        ("session manager refresh token", "mixed"),
        ("parseConfig loadSettings v2", "mixed"),
        ("", "mixed"),
    ],
)
def test_query_shape(query: str, kind: str) -> None:
    assert analyze_query(query).kind == kind


def test_edge_punctuation_is_trimmed_from_tokens() -> None:
    assert analyze_query('(where is "auth")?').tokens == ("where", "is", "auth")


# ---------------------------------------------------------- keyword queries


def test_stopwords_are_dropped_from_natural_language_questions() -> None:
    query = "where do we handle expired logins?"
    assert keyword_query(query, analyze_query(query)) == "we handle expired logins"
    kept = keyword_query(query, analyze_query(query), drop_stopwords=False)
    assert kept == "where do we handle expired logins"


def test_a_query_of_only_stopwords_is_kept_whole() -> None:
    assert keyword_query("what is it", analyze_query("what is it")) == "what is it"


def test_identifiers_are_also_searched_as_their_parts() -> None:
    shape = analyze_query("getUserById")
    assert keyword_query("getUserById", shape) == "getUserById get user by id"
    assert keyword_query("getUserById", shape, expand_identifiers=False) == "getUserById"


def test_expansion_does_not_repeat_terms_already_present() -> None:
    query = "user getUserById"
    assert keyword_query(query, analyze_query(query)).split().count("user") == 1


# ----------------------------------------------------------------- filters


def test_filters() -> None:
    plain = chunk(1, "src/a.ts")
    test_file = chunk(2, "tests/a.ts", tags="test")
    assert SearchFilters().allows(test_file) and not SearchFilters().active
    assert SearchFilters(path_prefix="src/").allows(plain)
    assert not SearchFilters(path_prefix="src/").allows(test_file)
    assert not SearchFilters(exclude_tags=frozenset({"test"})).allows(test_file)
    assert SearchFilters(languages=frozenset({"ts"})).allows(plain)
    assert not SearchFilters(languages=frozenset({"py"})).allows(plain)


# --------------------------------------------------------------- diversity


def test_overlap_is_measured_against_the_shorter_chunk() -> None:
    assert _overlap(chunk(1, start=1, end=10), chunk(2, start=5, end=12)) == pytest.approx(6 / 8)
    assert _overlap(chunk(1, start=1, end=10), chunk(2, start=20, end=30)) == 0.0
    assert _overlap(chunk(1, start=1, end=100), chunk(2, start=10, end=12)) == 1.0


def test_at_most_n_hits_per_file() -> None:
    chunks = {i: chunk(i, "big.ts", start=i * 20, end=i * 20 + 10) for i in range(1, 6)}
    chunks[9] = chunk(9, "other.ts")
    ranked = [(i, 1.0 / i) for i in (1, 2, 3, 4, 5, 9)]
    chosen = _diversify(ranked, chunks, SearchConfig(k=10, max_per_file=2))
    assert [i for i, _ in chosen] == [1, 2, 9]


def test_heavily_overlapping_chunks_of_one_file_collapse() -> None:
    chunks = {
        1: chunk(1, start=1, end=10),
        2: chunk(2, start=5, end=12),
        3: chunk(3, start=20, end=30),
    }
    ranked = [(1, 3.0), (2, 2.0), (3, 1.0)]
    assert [i for i, _ in _diversify(ranked, chunks, SearchConfig())] == [1, 3]
    kept = _diversify(ranked, chunks, SearchConfig(collapse_overlaps=False))
    assert [i for i, _ in kept] == [1, 2, 3]


def test_k_limits_the_number_of_results() -> None:
    chunks = {i: chunk(i, f"f{i}.ts") for i in range(1, 8)}
    ranked = [(i, 1.0) for i in range(1, 8)]
    assert len(_diversify(ranked, chunks, SearchConfig(k=3))) == 3
