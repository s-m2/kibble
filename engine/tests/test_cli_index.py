from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kibble_engine.cli import main
from kibble_engine.core import session as session_module
from kibble_engine.embedders import EmbedderConfig, OpenAICompatibleEmbedder
from kibble_engine.testing.fake_embedding_server import FakeEmbeddingServer
from kibble_engine.testing.helpers import init_repo, write

CONFIG = (
    '[embedder]\nbase_url = "http://fake.test/v1"\nmodel = "fake-embed"\n'
    "max_input_chars = 2000\nbatch_size = 16\nconcurrency = 1\n"
)


def prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, FakeEmbeddingServer]:
    repo = init_repo(
        tmp_path / "repo",
        {"a.txt": "validate the token here\n", "b.txt": "load settings from a file\n"},
        git=False,
    )
    (repo / ".kibble").mkdir()
    (repo / ".kibble" / "config.toml").write_text(CONFIG, encoding="utf-8")
    server = FakeEmbeddingServer(dim=32)

    def build(config: EmbedderConfig, env: Any = None) -> OpenAICompatibleEmbedder:
        return OpenAICompatibleEmbedder(config, client=server.client(), sleep=lambda s: None)

    monkeypatch.setattr(session_module, "build_embedder", build)
    return repo, server


def test_index_then_status_then_edit_then_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, server = prepare(tmp_path, monkeypatch)
    args = ["--repo", str(repo)]

    assert main(["status", *args]) == 1  # no index yet
    assert main(["index", *args]) == 0
    assert main(["status", *args]) == 0  # nothing pending

    write(repo, "a.txt", "completely different content now\n")
    assert main(["status", *args]) == 1  # status sees the pending change without fixing it
    assert main(["search", "different content", *args]) == 0  # search refreshes first
    assert main(["status", *args]) == 0


def test_missing_config_is_a_configuration_error(tmp_path: Path) -> None:
    repo = init_repo(tmp_path / "repo", {"a.txt": "x\n"}, git=False)
    assert main(["index", "--repo", str(repo)]) == 2


def test_drift_blocks_indexing_unless_overridden(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, server = prepare(tmp_path, monkeypatch)
    args = ["--repo", str(repo)]
    assert main(["index", *args]) == 0

    server.perturb = 0.5  # the endpoint now serves something different
    assert main(["index", *args]) == 3
    assert main(["index", *args, "--ignore-drift"]) == 0


def test_an_unreachable_endpoint_is_exit_code_1_and_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, server = prepare(tmp_path, monkeypatch)
    args = ["--repo", str(repo)]
    assert main(["index", *args]) == 0
    write(repo, "a.txt", "edited while the endpoint is down\n")

    server.fail_next(503, times=50)
    assert main(["index", *args]) == 1  # the probe fails before anything is touched
    server._script.clear()
    assert main(["status", *args]) == 1  # the edit is still pending
    assert main(["index", *args]) == 0  # and completes once the endpoint is back
    assert main(["status", *args]) == 0
