"""Open everything needed to index and search one repository.

Shared by the CLI and (later) the HTTP API: load `.env` and config, build and probe the
embedder, open the index with that embedder's identity, record or verify the drift
canary, and construct the refresher.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from kibble_engine.core.refresh import Refresher
from kibble_engine.core.search import HybridSearcher
from kibble_engine.dotenv import load_dotenv_file
from kibble_engine.embedders import (
    CanaryResult,
    EmbedderConfig,
    OpenAICompatibleEmbedder,
    build_embedder,
    check_canary,
    load_config,
)
from kibble_engine.parsing import DEFAULT_MAX_CHARS
from kibble_engine.store.sqlite import SqliteStore

INDEX_RELATIVE_PATH = Path(".kibble") / "index.db"


def environment_for(repo: Path, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """`.env` in the repo root underneath the real environment (the environment wins)."""
    real = os.environ if environ is None else environ
    return {**load_dotenv_file(repo / ".env"), **real}


@dataclass
class Session:
    root: Path
    config: EmbedderConfig
    embedder: OpenAICompatibleEmbedder
    store: SqliteStore
    refresher: Refresher
    canary: CanaryResult
    searcher: HybridSearcher

    def close(self) -> None:
        self.store.close()
        self.embedder.close()

    def __enter__(self) -> Session:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def open_session(
    repo: Path,
    *,
    config_path: Path | None = None,
    config: EmbedderConfig | None = None,
    environ: Mapping[str, str] | None = None,
    embedder: OpenAICompatibleEmbedder | None = None,
    use_git: bool = True,
) -> Session:
    """Raises ConfigError, EmbedderError or StoreError with a message fit to print."""
    root = repo.resolve()
    env = environment_for(root, environ)
    cfg = config if config is not None else load_config(config_path, env=env, cwd=root)
    emb = embedder if embedder is not None else build_embedder(cfg, env)
    try:
        emb.probe()
        store = SqliteStore.open(
            root / INDEX_RELATIVE_PATH,
            embedder_id=emb.info.embedder_id,
            embed_dim=emb.info.dim,
        )
        try:
            canary = check_canary(emb, store)
        except BaseException:
            store.close()
            raise
    except BaseException:
        emb.close()
        raise
    refresher = Refresher(
        root,
        store,
        emb,
        use_git=use_git,
        max_chars=min(DEFAULT_MAX_CHARS, cfg.max_input_chars),
    )
    searcher = HybridSearcher(store, emb, refresher)
    return Session(root, cfg, emb, store, refresher, canary, searcher)
