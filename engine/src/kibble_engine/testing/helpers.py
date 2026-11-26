"""Shared test helpers: throwaway repositories and an embedder wired to the fake server."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from kibble_engine.embedders import EmbedderConfig, OpenAICompatibleEmbedder
from kibble_engine.store.sqlite import SqliteStore
from kibble_engine.testing.fake_embedding_server import FakeEmbeddingServer


def git_available() -> bool:
    return shutil.which("git") is not None


def age_files(root: Path, seconds: float = 3600) -> None:
    """Backdate every file's mtime, so tests start on the sweep's fast path (not 'racy')."""
    stamp = time.time() - seconds
    for path in root.rglob("*"):
        if path.is_file() and ".git" not in path.parts:
            os.utime(path, (stamp, stamp))


def init_repo(
    root: Path, files: Mapping[str, str | bytes], *, git: bool = True, age_s: float = 3600
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if git:
        subprocess.run(["git", "init", "-q", str(root)], check=True)
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
    age_files(root, age_s)
    return root


def write(root: Path, rel: str, content: str | bytes, *, age_s: float | None = None) -> None:
    """Create or overwrite a file; optionally backdate it."""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    if age_s is not None:
        stamp = time.time() - age_s
        os.utime(path, (stamp, stamp))


def make_embedder(server: FakeEmbeddingServer, **overrides: Any) -> OpenAICompatibleEmbedder:
    settings: dict[str, Any] = {
        "base_url": "http://fake.test/v1",
        "model": server.model,
        "concurrency": 1,
        "max_retries": 1,
        **overrides,
    }
    embedder = OpenAICompatibleEmbedder(
        EmbedderConfig.from_mapping(settings), client=server.client(), sleep=lambda s: None
    )
    embedder.probe()
    return embedder


def open_store(root: Path, embedder: OpenAICompatibleEmbedder) -> SqliteStore:
    return SqliteStore.open(
        root / ".kibble" / "index.db",
        embedder_id=embedder.info.embedder_id,
        embed_dim=embedder.info.dim,
        vector_backend="python",
    )


def grammars_available() -> bool:
    try:
        import tree_sitter
        import tree_sitter_typescript  # noqa: F401
    except ImportError:
        return False
    return bool(tree_sitter)
