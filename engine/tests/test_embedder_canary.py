from __future__ import annotations

from typing import Any

from kibble_engine.embedders import EmbedderConfig, OpenAICompatibleEmbedder, check_canary
from kibble_engine.testing.fake_embedding_server import FakeEmbeddingServer


class DictMeta:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}

    def get_meta(self, key: str) -> str | None:
        return self.data.get(key)

    def set_meta(self, key: str, value: str) -> None:
        self.data[key] = value


def embedder_for(server: FakeEmbeddingServer, **overrides: Any) -> OpenAICompatibleEmbedder:
    config = EmbedderConfig.from_mapping(
        {"base_url": "http://fake.test/v1", "model": server.model, "concurrency": 1, **overrides}
    )
    return OpenAICompatibleEmbedder(config, client=server.client(), sleep=lambda s: None)


def test_first_run_records_and_later_runs_match() -> None:
    server, meta = FakeEmbeddingServer(dim=64), DictMeta()
    first = check_canary(embedder_for(server), meta)
    assert first.status == "recorded" and first.similarity is None
    assert len(meta.data) == 1

    again = check_canary(embedder_for(server), meta)
    assert again.status == "ok" and again.similarity is not None and again.similarity > 0.9999


def test_a_changed_endpoint_is_reported_as_drift() -> None:
    server, meta = FakeEmbeddingServer(dim=64), DictMeta()
    check_canary(embedder_for(server), meta)

    server.perturb = 0.5  # a different build or quantization
    result = check_canary(embedder_for(server), meta)
    assert result.status == "drift"
    assert result.similarity is not None and result.similarity < 0.999
    assert "re-embed" in result.message


def test_negligible_numeric_noise_is_not_drift() -> None:
    server, meta = FakeEmbeddingServer(dim=64), DictMeta()
    check_canary(embedder_for(server), meta)
    server.perturb = 1e-6
    assert check_canary(embedder_for(server), meta).status == "ok"


def test_each_embedder_identity_has_its_own_reference() -> None:
    server, meta = FakeEmbeddingServer(dim=64), DictMeta()
    check_canary(embedder_for(server), meta)
    other = check_canary(embedder_for(server, doc_prefix="doc: "), meta)
    assert other.status == "recorded"  # a different prefix is a different vector space
    assert len(meta.data) == 2
