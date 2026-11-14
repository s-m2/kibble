"""Embedders. One OpenAI-compatible implementation covers hosted, LAN and local servers."""

from __future__ import annotations

from collections.abc import Mapping

from kibble_engine.embedders.base import (
    Embedder,
    EmbedderInfo,
    EmbedderStats,
    ProbeResult,
    compute_embedder_id,
)
from kibble_engine.embedders.canary import CanaryResult, check_canary
from kibble_engine.embedders.config import (
    EmbedderConfig,
    cleartext_key_warning,
    load_config,
    resolve_api_key,
)
from kibble_engine.embedders.doctor import Check, run_doctor
from kibble_engine.embedders.errors import (
    ConfigError,
    EmbedderAuthError,
    EmbedderConnectionError,
    EmbedderCreditsError,
    EmbedderDimensionError,
    EmbedderError,
    EmbedderNotFoundError,
    EmbedderRateLimitError,
    EmbedderRequestError,
    EmbedderResponseError,
    EmbedderServerError,
)
from kibble_engine.embedders.openai_compatible import OpenAICompatibleEmbedder


def build_embedder(
    config: EmbedderConfig, env: Mapping[str, str] | None = None
) -> OpenAICompatibleEmbedder:
    """Construct the embedder, resolving the API key from the environment."""
    return OpenAICompatibleEmbedder(config, api_key=resolve_api_key(config, env))


__all__ = [
    "CanaryResult",
    "Check",
    "ConfigError",
    "Embedder",
    "EmbedderAuthError",
    "EmbedderConfig",
    "EmbedderConnectionError",
    "EmbedderCreditsError",
    "EmbedderDimensionError",
    "EmbedderError",
    "EmbedderInfo",
    "EmbedderNotFoundError",
    "EmbedderRateLimitError",
    "EmbedderRequestError",
    "EmbedderResponseError",
    "EmbedderServerError",
    "EmbedderStats",
    "OpenAICompatibleEmbedder",
    "ProbeResult",
    "build_embedder",
    "check_canary",
    "cleartext_key_warning",
    "compute_embedder_id",
    "load_config",
    "resolve_api_key",
    "run_doctor",
]
