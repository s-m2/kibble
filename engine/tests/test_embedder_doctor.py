from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import kibble_engine.embedders as embedders_module
from kibble_engine.cli import main
from kibble_engine.embedders import EmbedderConfig, OpenAICompatibleEmbedder, run_doctor
from kibble_engine.testing.fake_embedding_server import FakeEmbeddingServer


def setup(
    server: FakeEmbeddingServer, *, api_key: str | None = None, **overrides: Any
) -> tuple[OpenAICompatibleEmbedder, EmbedderConfig]:
    settings: dict[str, Any] = {
        "base_url": "http://localhost:8080/v1",
        "model": server.model,
        "concurrency": 1,
        "batch_size": 16,
        "max_input_chars": 1000,
        **overrides,
    }
    config = EmbedderConfig.from_mapping(settings)
    embedder = OpenAICompatibleEmbedder(
        config, api_key=api_key, client=server.client(), sleep=lambda s: None
    )
    return embedder, config


def statuses(
    embedder: OpenAICompatibleEmbedder, config: EmbedderConfig, *, has_key: bool = False
) -> dict[str, str]:
    checks = run_doctor(embedder, config, has_key=has_key)
    return {c.name: c.status for c in checks}


def test_healthy_endpoint_passes_every_check() -> None:
    embedder, config = setup(FakeEmbeddingServer(dim=128))
    checks = run_doctor(embedder, config, has_key=False)
    assert {c.name: c.status for c in checks} == {
        "endpoint": "ok",
        "batching": "ok",
        "input length": "ok",
        "stability": "ok",
        "retrieval": "ok",
    }
    assert "128-dim" in checks[0].detail
    assert embedder.stats.requests == 7  # users on quota-limited tiers care about this


def test_unstable_backend_is_flagged() -> None:
    server = FakeEmbeddingServer(dim=128)
    server.jitter = 0.5
    embedder, config = setup(server)
    checks = {c.name: c for c in run_doctor(embedder, config, has_key=False)}
    assert checks["stability"].status == "warn"
    assert "different vectors" in checks["stability"].detail


def test_prefixes_that_hurt_retrieval_are_called_out() -> None:
    server = FakeEmbeddingServer(dim=128)
    server.prefix_breaks_model = True  # e.g. the endpoint already adds its own prefixes
    embedder, config = setup(server, doc_prefix="passage: ", query_prefix="query: ")
    checks = {c.name: c for c in run_doctor(embedder, config, has_key=False)}
    assert checks["retrieval"].status == "warn"
    assert "without them" in checks["retrieval"].detail

    clean, config2 = setup(
        FakeEmbeddingServer(dim=128), doc_prefix="passage: ", query_prefix="query: "
    )
    assert statuses(clean, config2)["retrieval"] == "ok"


def test_a_model_that_cannot_rank_the_sanity_queries_is_flagged() -> None:
    server = FakeEmbeddingServer(dim=128)
    server.perturb = 50.0  # noise swamps the signal
    embedder, config = setup(server)
    checks = {c.name: c for c in run_doctor(embedder, config, has_key=False)}
    assert checks["retrieval"].status == "warn"
    assert "sanity queries" in checks["retrieval"].detail


def test_unreachable_or_unauthorised_endpoint_stops_at_the_first_failure() -> None:
    server = FakeEmbeddingServer(api_key="right")
    embedder, config = setup(server, api_key="wrong")
    checks = run_doctor(embedder, config, has_key=True)
    assert [(c.name, c.status) for c in checks] == [("endpoint", "fail")]
    assert "401" in checks[0].detail


def test_server_batch_limit_is_reported_as_a_warning_with_the_fix() -> None:
    embedder, config = setup(FakeEmbeddingServer(max_batch=8), batch_size=32)
    checks = {c.name: c for c in run_doctor(embedder, config, has_key=False)}
    assert checks["batching"].status == "warn"
    assert "embedder.batch_size" in checks["batching"].detail


def test_server_length_limit_is_reported_as_a_warning_with_the_fix() -> None:
    embedder, config = setup(FakeEmbeddingServer(max_input_chars=300), max_input_chars=2000)
    checks = {c.name: c for c in run_doctor(embedder, config, has_key=False)}
    assert checks["input length"].status == "warn"
    assert "max_input_chars" in checks["input length"].detail


def test_missing_prefixes_for_known_model_families_are_flagged() -> None:
    server = FakeEmbeddingServer(model="nomic-embed-text")
    embedder, config = setup(server)
    result = statuses(embedder, config)
    assert result["prefixes"] == "warn"

    configured, config2 = setup(
        server, doc_prefix="search_document: ", query_prefix="search_query: "
    )
    assert "prefixes" not in statuses(configured, config2)


def test_nemotron_embed_models_are_flagged_without_prefixes() -> None:
    server = FakeEmbeddingServer(model="nvidia/Nemotron-3-Embed-8B-BF16")
    embedder, config = setup(server)
    checks = {c.name: c for c in run_doctor(embedder, config, has_key=False)}
    assert checks["prefixes"].status == "warn"
    assert '"query: "' in checks["prefixes"].detail

    configured, config2 = setup(server, doc_prefix="passage: ", query_prefix="query: ")
    assert "prefixes" not in statuses(configured, config2)


def test_key_over_cleartext_lan_http_is_flagged() -> None:
    server = FakeEmbeddingServer(api_key="k")
    embedder, config = setup(server, api_key="k", base_url="http://gpu-box:8080/v1")
    assert statuses(embedder, config, has_key=True)["transport"] == "warn"


# ----------------------------------------------------------------------- CLI


def test_doctor_command_reports_config_errors_with_exit_code_2(tmp_path: Path) -> None:
    assert main(["doctor", "--config", str(tmp_path / "missing.toml")]) == 2
    bad = tmp_path / "bad.toml"
    bad.write_text('[embedder]\nmodel = "m"\n', encoding="utf-8")  # no base_url
    assert main(["doctor", "--config", str(bad)]) == 2


def test_doctor_command_exit_code_reflects_the_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[embedder]\nbase_url = "http://localhost:8080/v1"\nmodel = "fake-embed"\n'
        "max_input_chars = 1000\nbatch_size = 16\n",
        encoding="utf-8",
    )

    def install(server: FakeEmbeddingServer) -> None:
        def build(config: EmbedderConfig, env: Any = None) -> OpenAICompatibleEmbedder:
            return OpenAICompatibleEmbedder(config, client=server.client(), sleep=lambda s: None)

        monkeypatch.setattr(embedders_module, "build_embedder", build)

    install(FakeEmbeddingServer())
    assert main(["doctor", "--config", str(config_path)]) == 0

    broken = FakeEmbeddingServer()
    broken.reject_all_with = 400
    install(broken)
    assert main(["doctor", "--config", str(config_path)]) == 1
