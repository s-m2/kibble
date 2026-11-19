from __future__ import annotations

from pathlib import Path

from kibble_engine.core.session import INDEX_RELATIVE_PATH, Session, environment_for, open_session
from kibble_engine.embedders import (
    EmbedderConfig,
    OpenAICompatibleEmbedder,
    load_config,
    resolve_api_key,
)
from kibble_engine.testing.fake_embedding_server import FakeEmbeddingServer
from kibble_engine.testing.helpers import init_repo


def open_with(repo: Path, server: FakeEmbeddingServer) -> Session:
    config = EmbedderConfig.from_mapping(
        {"base_url": "http://fake.test/v1", "model": server.model, "concurrency": 1}
    )
    embedder = OpenAICompatibleEmbedder(config, client=server.client(), sleep=lambda s: None)
    return open_session(repo, config=config, embedder=embedder, use_git=False)


def test_session_creates_the_index_and_records_a_canary(tmp_path: Path) -> None:
    init_repo(tmp_path, {"a.txt": "hello world\n"}, git=False)
    with open_with(tmp_path, FakeEmbeddingServer(dim=32)) as session:
        assert session.canary.status == "recorded"
        assert session.refresher.refresh().refreshed == ["a.txt"]
    assert (tmp_path / INDEX_RELATIVE_PATH).exists()


def test_reopening_verifies_the_canary_and_keeps_the_index(tmp_path: Path) -> None:
    init_repo(tmp_path, {"a.txt": "hello world\n"}, git=False)
    server = FakeEmbeddingServer(dim=32)
    with open_with(tmp_path, server) as first:
        first.refresher.refresh()
    with open_with(tmp_path, server) as second:
        assert second.canary.status == "ok"
        assert second.refresher.refresh().unchanged == 1  # the index persisted


def test_a_changed_endpoint_is_reported_as_drift(tmp_path: Path) -> None:
    init_repo(tmp_path, {"a.txt": "hello world\n"}, git=False)
    server = FakeEmbeddingServer(dim=32)
    open_with(tmp_path, server).close()
    server.perturb = 0.5
    with open_with(tmp_path, server) as session:
        assert session.canary.status == "drift" and "re-embed" in session.canary.message


def test_api_key_can_come_from_the_repos_dotenv_file(tmp_path: Path) -> None:
    (tmp_path / ".kibble").mkdir()
    (tmp_path / ".kibble" / "config.toml").write_text(
        '[embedder]\nbase_url = "https://openrouter.ai/api/v1"\nmodel = "m"\n'
        'api_key_env = "OPENROUTER_API_KEY"\n',
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=sk-from-dotenv\n", encoding="utf-8")

    env = environment_for(tmp_path, {})
    config = load_config(cwd=tmp_path, env=env)
    assert resolve_api_key(config, env) == "sk-from-dotenv"

    exported = environment_for(tmp_path, {"OPENROUTER_API_KEY": "sk-exported"})
    assert resolve_api_key(config, exported) == "sk-exported"  # the real environment wins
