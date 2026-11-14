"""The OpenAI-compatible embedder against the in-process fake server."""

from __future__ import annotations

import math
from typing import Any

import pytest

from kibble_engine.embedders import (
    EmbedderAuthError,
    EmbedderConfig,
    EmbedderConnectionError,
    EmbedderCreditsError,
    EmbedderDimensionError,
    EmbedderError,
    EmbedderNotFoundError,
    EmbedderRateLimitError,
    EmbedderRequestError,
    EmbedderResponseError,
    EmbedderServerError,
    OpenAICompatibleEmbedder,
)
from kibble_engine.embedders.http import httpx
from kibble_engine.testing.fake_embedding_server import FakeEmbeddingServer


def make(
    server: FakeEmbeddingServer, **overrides: Any
) -> tuple[OpenAICompatibleEmbedder, list[float]]:
    """An embedder wired to `server`, with sleeping replaced by a recorder."""
    settings: dict[str, Any] = {
        "base_url": "http://fake.test/v1",
        "model": server.model,
        "concurrency": 1,
        **overrides,
    }
    sleeps: list[float] = []
    embedder = OpenAICompatibleEmbedder(
        EmbedderConfig.from_mapping(settings),
        api_key=server.api_key,
        client=server.client(),
        sleep=sleeps.append,
    )
    return embedder, sleeps


def expected(texts: list[str], dim: int = 16, prefix: str = "") -> list[list[float]]:
    return [FakeEmbeddingServer.embed_text(prefix + t, dim) for t in texts]


# ------------------------------------------------------------------- basics


def test_probe_learns_the_dimension_and_info_requires_it() -> None:
    server = FakeEmbeddingServer(dim=24)
    embedder, _ = make(server)
    with pytest.raises(EmbedderError, match="probe"):
        _ = embedder.info
    result = embedder.probe()
    assert result.dim == 24 and result.latency_s >= 0
    assert embedder.info.dim == 24 and embedder.info.model == "fake-embed"
    assert embedder.info.host == "fake.test"


def test_embedder_id_ignores_base_url_and_query_prefix_but_not_doc_prefix() -> None:
    server = FakeEmbeddingServer()
    ids = {}
    for label, overrides in {
        "base": {},
        "other_host": {"base_url": "http://gpu-box:9/v1"},
        "query_prefix": {"query_prefix": "query: "},
        "doc_prefix": {"doc_prefix": "doc: "},
    }.items():
        embedder, _ = make(server, **overrides)
        embedder.probe()
        ids[label] = embedder.info.embedder_id
    assert ids["base"] == ids["other_host"] == ids["query_prefix"]
    assert ids["doc_prefix"] != ids["base"]


def test_documents_are_batched_and_returned_in_order() -> None:
    server = FakeEmbeddingServer()
    embedder, _ = make(server, batch_size=4)
    texts = [f"document number {i}" for i in range(10)]
    assert embedder.embed_documents(texts) == expected(texts)
    assert [len(r["input"]) for r in server.requests] == [4, 4, 2]


def test_order_is_restored_from_item_index_not_response_order() -> None:
    server = FakeEmbeddingServer()
    server.shuffle = True
    embedder, _ = make(server, batch_size=8)
    texts = [f"text {i}" for i in range(8)]
    assert embedder.embed_documents(texts) == expected(texts)


def test_concurrent_batches_keep_input_order() -> None:
    server = FakeEmbeddingServer()
    embedder, _ = make(server, batch_size=5, concurrency=4)
    texts = [f"concurrent text {i}" for i in range(25)]
    assert embedder.embed_documents(texts) == expected(texts)
    assert sorted(len(r["input"]) for r in server.requests) == [5] * 5


def test_progress_callback_reports_running_totals() -> None:
    server = FakeEmbeddingServer()
    embedder, _ = make(server, batch_size=4)
    seen: list[tuple[int, int]] = []
    embedder.embed_documents(
        [f"t{i}" for i in range(10)], on_progress=lambda done, total: seen.append((done, total))
    )
    assert seen == [(4, 10), (8, 10), (10, 10)]


def test_documents_and_queries_use_their_own_prefixes() -> None:
    server = FakeEmbeddingServer()
    embedder, _ = make(server, doc_prefix="search_document: ", query_prefix="search_query: ")
    embedder.embed_documents(["alpha"])
    embedder.embed_query("beta")
    assert server.requests[0]["input"] == ["search_document: alpha"]
    assert server.requests[1]["input"] == ["search_query: beta"]


def test_long_input_is_truncated_but_the_prefix_survives() -> None:
    server = FakeEmbeddingServer()
    embedder, _ = make(server, doc_prefix="doc: ", max_input_chars=100)
    embedder.embed_documents(["x" * 500])
    sent = server.requests[0]["input"][0]
    assert len(sent) == 100 and sent.startswith("doc: ")
    assert embedder.stats.truncations == 1


def test_empty_input_is_rejected_before_any_request() -> None:
    server = FakeEmbeddingServer()
    embedder, _ = make(server)
    with pytest.raises(ValueError, match="empty"):
        embedder.embed_documents(["fine", "   "])
    assert embedder.embed_documents([]) == []
    assert server.requests == []


def test_bearer_token_is_sent_only_when_configured() -> None:
    keyed = FakeEmbeddingServer(api_key="s3cret")
    make(keyed)[0].embed_query("hello")
    assert keyed.auth_headers == ["Bearer s3cret"]
    open_server = FakeEmbeddingServer()
    make(open_server)[0].embed_query("hello")
    assert open_server.auth_headers == [None]


# --------------------------------------------------------- hard failures


def test_auth_failure_names_the_key_variable() -> None:
    server = FakeEmbeddingServer(api_key="right")
    embedder = OpenAICompatibleEmbedder(
        EmbedderConfig.from_mapping(
            {"base_url": "http://fake.test/v1", "model": "fake-embed", "api_key_env": "MY_KEY"}
        ),
        api_key="wrong",
        client=server.client(),
        sleep=lambda s: None,
    )
    with pytest.raises(EmbedderAuthError, match=r"401.*MY_KEY"):
        embedder.embed_query("hi")


def test_credits_and_not_found_are_distinct_errors() -> None:
    server = FakeEmbeddingServer()
    embedder, sleeps = make(server)
    server.fail_next(402, message="insufficient credits")
    with pytest.raises(EmbedderCreditsError, match="insufficient credits"):
        embedder.embed_query("hi")
    wrong_path, _ = make(FakeEmbeddingServer(path="/embeddings"))
    with pytest.raises(EmbedderNotFoundError, match=r"base_url.*/v1"):
        wrong_path.embed_query("hi")
    wrong_model, _ = make(FakeEmbeddingServer(), model="nope")
    with pytest.raises(EmbedderNotFoundError, match="'nope'"):
        wrong_model.embed_query("hi")
    assert sleeps == []  # hard failures are never retried


# --------------------------------------------------------- transient failures


@pytest.mark.parametrize("status", [408, 524, 529])
def test_provider_timeouts_and_overload_are_retried(status: int) -> None:
    server = FakeEmbeddingServer()
    embedder, sleeps = make(server)
    server.fail_next(status)
    assert embedder.embed_query("hello")
    assert len(sleeps) == 1


def test_rate_limit_message_explains_quotas() -> None:
    server = FakeEmbeddingServer()
    embedder, _ = make(server, max_retries=0)
    server.fail_next(429)
    with pytest.raises(EmbedderRateLimitError, match=r"requests_per_minute.*daily"):
        embedder.embed_query("hello")


def test_requests_are_paced_to_the_configured_quota() -> None:
    server = FakeEmbeddingServer()
    now = [100.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    embedder = OpenAICompatibleEmbedder(
        EmbedderConfig.from_mapping(
            {
                "base_url": "http://fake.test/v1",
                "model": server.model,
                "concurrency": 1,
                "batch_size": 1,
                "requests_per_minute": 30,  # one request every 2 seconds
            }
        ),
        client=server.client(),
        sleep=sleep,
        clock=lambda: now[0],
    )
    embedder.embed_documents(["a", "b", "c", "d"])
    assert sleeps == [2.0, 2.0, 2.0]  # the first request goes out immediately


def test_pacing_is_shared_with_sibling_embedders() -> None:
    server = FakeEmbeddingServer()
    now = [0.0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    base = OpenAICompatibleEmbedder(
        EmbedderConfig.from_mapping(
            {"base_url": "http://fake.test/v1", "model": server.model, "requests_per_minute": 60}
        ),
        client=server.client(),
        sleep=sleep,
        clock=lambda: now[0],
    )
    sibling = base.with_overrides(doc_prefix="doc: ")
    base.embed_query("one")
    sibling.embed_documents(["two"])
    assert now[0] == 1.0  # the sibling waited for the shared slot
    assert server.requests[1]["input"] == ["doc: two"]


# ------------------------------------------------------------- query batching


def test_queries_are_batched_with_the_query_prefix() -> None:
    server = FakeEmbeddingServer()
    embedder, _ = make(server, batch_size=2, query_prefix="query: ", doc_prefix="doc: ")
    texts = ["alpha", "beta", "gamma"]
    assert embedder.embed_queries(texts) == expected(texts, prefix="query: ")
    assert [r["input"] for r in server.requests] == [
        ["query: alpha", "query: beta"],
        ["query: gamma"],
    ]


def test_transient_server_errors_are_retried_with_backoff() -> None:
    server = FakeEmbeddingServer()
    embedder, sleeps = make(server)
    server.fail_next(503, times=2)
    assert embedder.embed_query("hello") == expected(["hello"])[0]
    assert embedder.stats.retries == 2 and len(sleeps) == 2
    assert sleeps[1] > sleeps[0]  # exponential


def test_retry_after_header_is_honoured() -> None:
    server = FakeEmbeddingServer()
    embedder, sleeps = make(server)
    server.fail_next(429, headers={"Retry-After": "2"})
    embedder.embed_query("hello")
    assert sleeps == [2.0]


def test_persistent_failures_give_up_after_max_retries() -> None:
    server = FakeEmbeddingServer()
    embedder, sleeps = make(server, max_retries=2)
    server.fail_next(503, times=10)
    with pytest.raises(EmbedderServerError, match="503"):
        embedder.embed_query("hello")
    assert len(sleeps) == 2 and embedder.stats.requests == 3

    embedder, _ = make(server, max_retries=1)
    server._script.clear()
    server.fail_next(429, times=10)
    with pytest.raises(EmbedderRateLimitError, match="concurrency"):
        embedder.embed_query("hello")


def test_connection_failures_are_retried_then_reported() -> None:
    server = FakeEmbeddingServer()
    embedder, sleeps = make(server, max_retries=2)
    server.drop_next(1)
    assert embedder.embed_query("hello")  # recovers after one dropped connection
    assert len(sleeps) == 1

    server.drop_next(10)
    with pytest.raises(EmbedderConnectionError, match=r"could not connect to http://fake.test/v1"):
        embedder.embed_query("hello")


def test_timeouts_get_a_hint_about_model_loading() -> None:
    server = FakeEmbeddingServer()
    embedder, _ = make(server, max_retries=0)

    def timeout(request: Any) -> Any:
        raise httpx.TimeoutException("slow", request=request)

    server._script.append(timeout)
    with pytest.raises(EmbedderConnectionError, match=r"timed out.*loads the model"):
        embedder.embed_query("hello")


# ------------------------------------------------------ adapting to servers


def test_batches_over_the_servers_limit_are_split_automatically() -> None:
    server = FakeEmbeddingServer(max_batch=4)
    embedder, _ = make(server, batch_size=16)
    texts = [f"item {i}" for i in range(16)]
    assert embedder.embed_documents(texts) == expected(texts)
    assert embedder.stats.batch_splits >= 1
    assert max(server.accepted_batch_sizes) <= 4


def test_overlong_single_input_is_shrunk_until_it_fits() -> None:
    server = FakeEmbeddingServer(max_input_chars=200)
    embedder, _ = make(server, max_input_chars=2000)
    text = "word " * 180  # 900 chars
    assert len(embedder.embed_documents([text])) == 1
    assert embedder.stats.input_shrinks == 3  # 450 -> 225 -> 112
    assert len(server.requests[-1]["input"][0]) <= 200


def test_input_that_never_fits_fails_with_a_fix_in_the_message() -> None:
    server = FakeEmbeddingServer(max_input_chars=50)
    embedder, _ = make(server, max_input_chars=2000)
    with pytest.raises(EmbedderRequestError, match=r"max_input_chars.*max_chars"):
        embedder.embed_documents(["word " * 180])
    assert len(server.requests) == 4  # the original plus three shrinking attempts


def test_unsupported_request_fails_fast_instead_of_thrashing() -> None:
    server = FakeEmbeddingServer()
    server.reject_all_with = 400
    embedder, _ = make(server, batch_size=8)
    with pytest.raises(EmbedderRequestError, match="unsupported request"):
        embedder.embed_documents([f"short {i}" for i in range(8)])
    assert len(server.requests) <= 5  # 8 -> 4 -> 2 -> 1, then give up


# ------------------------------------------------------ malformed responses


def test_malformed_responses_raise_response_errors() -> None:
    server = FakeEmbeddingServer()
    embedder, _ = make(server)
    server.wrong_count = True
    with pytest.raises(EmbedderResponseError, match="1 embeddings for 2 inputs"):
        embedder.embed_documents(["a", "b"])
    server.wrong_count = False

    server.respond_next(httpx.Response(200, text="<html>not json</html>"))
    with pytest.raises(EmbedderResponseError, match="non-JSON"):
        embedder.embed_query("a")

    server.respond_next(httpx.Response(200, json={"data": [{"index": 0, "embedding": "AAECAw=="}]}))
    with pytest.raises(EmbedderResponseError, match="encoding_format"):
        embedder.embed_query("a")


# --------------------------------------------------------------- dimensions


def test_dimensions_are_requested_and_verified() -> None:
    honouring = FakeEmbeddingServer(dim=32)
    embedder, _ = make(honouring, dimensions=8)
    assert embedder.probe().dim == 8
    assert honouring.requests[0]["dimensions"] == 8

    ignoring = FakeEmbeddingServer(dim=32, honor_dimensions=False)
    embedder, _ = make(ignoring, dimensions=8)
    with pytest.raises(EmbedderDimensionError, match=r"dimensions=8.*ignores"):
        embedder.probe()


def test_ignored_dimensions_error_points_at_slice_dimensions() -> None:
    server = FakeEmbeddingServer(dim=32, honor_dimensions=False)
    embedder, _ = make(server, dimensions=8)
    with pytest.raises(EmbedderDimensionError, match="slice_dimensions"):
        embedder.probe()


def test_slice_dimensions_keeps_the_first_n_dims_and_renormalises() -> None:
    server = FakeEmbeddingServer(dim=16)
    embedder, _ = make(server, slice_dimensions=4)
    assert embedder.probe().dim == 4
    assert "dimensions" not in server.requests[0]  # slicing is client-side only

    [vector] = embedder.embed_documents(["validate token"])
    full = FakeEmbeddingServer.embed_text("validate token", 16)[:4]
    norm = math.sqrt(sum(x * x for x in full))
    assert vector == pytest.approx([x / norm for x in full])
    assert math.sqrt(sum(x * x for x in vector)) == pytest.approx(1.0)
    assert len(embedder.embed_query("validate token")) == 4


def test_slicing_changes_the_embedder_id_and_cannot_exceed_the_model_size() -> None:
    server = FakeEmbeddingServer(dim=16)
    plain, _ = make(server)
    sliced, _ = make(server, slice_dimensions=8)
    plain.probe()
    sliced.probe()
    assert plain.info.embedder_id != sliced.info.embedder_id
    assert sliced.info.dim == 8

    too_big, _ = make(server, slice_dimensions=64)
    with pytest.raises(EmbedderDimensionError, match="can't be sliced"):
        too_big.probe()


def test_dimension_drift_after_probe_is_caught() -> None:
    server = FakeEmbeddingServer(dim=16)
    embedder, _ = make(server)
    embedder.probe()
    server.dim = 32
    with pytest.raises(EmbedderDimensionError, match="may have changed"):
        embedder.embed_query("hello")


def test_probe_failure_is_a_clean_error() -> None:
    server = FakeEmbeddingServer()
    server.reject_all_with = 400
    embedder, _ = make(server)
    with pytest.raises(EmbedderRequestError, match="tiny test input"):
        embedder.probe()
