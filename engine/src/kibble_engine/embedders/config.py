"""Embedder configuration.

Where it comes from, lowest to highest precedence:

1. `[embedder]` table in `.kibble/config.toml` (or a path you pass)
2. environment variables named `KIBBLE_EMBED_<FIELD>` (e.g. KIBBLE_EMBED_MODEL)

Secrets are never read from the file: `api_key_env` names the environment variable
that holds the key. There is deliberately **no default endpoint or model**: chunk
text must never go to a hosted provider unless someone chose it.
"""

from __future__ import annotations

import ipaddress
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from kibble_engine.embedders.errors import ConfigError

# field name -> kind. "?" means optional (None / empty string -> None).
_KINDS: dict[str, str] = {
    "base_url": "str",
    "model": "str",
    "api_key_env": "str?",
    "batch_size": "int",
    "concurrency": "int",
    "requests_per_minute": "int?",
    "max_input_chars": "int",
    "timeout_s": "float",
    "probe_timeout_s": "float",
    "max_retries": "int",
    "backoff_base_s": "float",
    "dimensions": "int?",
    "slice_dimensions": "int?",
    "doc_prefix": "str",
    "query_prefix": "str",
    "verify_tls": "bool",
    "ca_bundle": "str?",
}
_REQUIRED = ("base_url", "model")
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


@dataclass(frozen=True, slots=True)
class EmbedderConfig:
    base_url: str
    model: str
    api_key_env: str | None = None
    batch_size: int = 64
    concurrency: int = 4
    requests_per_minute: int | None = None  # self-throttle to stay under a hosted quota
    max_input_chars: int = 8000
    timeout_s: float = 30.0
    probe_timeout_s: float = 120.0  # first request to a local server may load the model
    max_retries: int = 4
    backoff_base_s: float = 1.0
    dimensions: int | None = None  # ask the server to truncate vectors (if it supports it)
    slice_dimensions: int | None = None  # keep the first N dims client-side, then re-normalise
    doc_prefix: str = ""  # e.g. "search_document: " for models that need it
    query_prefix: str = ""  # e.g. "search_query: "
    verify_tls: bool = True
    ca_bundle: str | None = None  # path to a CA bundle for LAN servers with private CAs

    def __post_init__(self) -> None:
        url = self.base_url.strip().rstrip("/")
        if not url.startswith(("http://", "https://")) or not urlsplit(url).hostname:
            raise ConfigError(f"embedder.base_url must be an http(s) URL, got {self.base_url!r}")
        object.__setattr__(self, "base_url", url)
        if not self.model.strip():
            raise ConfigError("embedder.model must not be empty")
        _check_range("batch_size", self.batch_size, 1, 2048)
        _check_range("concurrency", self.concurrency, 1, 64)
        _check_range("max_input_chars", self.max_input_chars, 64, 1_000_000)
        if self.requests_per_minute is not None:
            _check_range("requests_per_minute", self.requests_per_minute, 1, 100_000)
        _check_range("max_retries", self.max_retries, 0, 10)
        if self.timeout_s <= 0 or self.probe_timeout_s <= 0:
            raise ConfigError("embedder.timeout_s and probe_timeout_s must be positive")
        if self.backoff_base_s < 0:
            raise ConfigError("embedder.backoff_base_s must not be negative")
        if self.dimensions is not None and self.dimensions < 1:
            raise ConfigError("embedder.dimensions must be positive")
        if self.slice_dimensions is not None and self.slice_dimensions < 1:
            raise ConfigError("embedder.slice_dimensions must be positive")
        if self.dimensions is not None and self.slice_dimensions is not None:
            raise ConfigError(
                "embedder.dimensions (asks the server to truncate) and slice_dimensions "
                "(the client truncates) are alternatives; set only one"
            )
        if self.ca_bundle and not self.verify_tls:
            raise ConfigError("embedder.ca_bundle and verify_tls=false contradict each other")

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/embeddings"

    @property
    def host(self) -> str:
        parts = urlsplit(self.base_url)
        return f"{parts.hostname}:{parts.port}" if parts.port else str(parts.hostname)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> EmbedderConfig:
        unknown = sorted(set(data) - set(_KINDS))
        if unknown:
            raise ConfigError(
                f"unknown embedder setting(s): {', '.join(unknown)}. "
                f"Known settings: {', '.join(_KINDS)}"
            )
        missing = [name for name in _REQUIRED if data.get(name) in (None, "")]
        if missing:
            raise ConfigError(
                f"missing required embedder setting(s): {', '.join(missing)}. Set them in "
                "[embedder] in .kibble/config.toml or as KIBBLE_EMBED_<NAME> environment "
                "variables (there is no default endpoint, by design)."
            )
        return cls(**{name: _coerce(name, value) for name, value in data.items()})


def _check_range(name: str, value: int, low: int, high: int) -> None:
    if not low <= value <= high:
        raise ConfigError(f"embedder.{name} must be between {low} and {high}, got {value}")


def _coerce(name: str, value: Any) -> Any:
    kind = _KINDS[name]
    optional = kind.endswith("?")
    base = kind.rstrip("?")
    if optional and (value is None or value == ""):
        return None
    try:
        if base == "str":
            if not isinstance(value, str):
                raise TypeError
            return value
        if base == "int":
            if isinstance(value, bool):
                raise TypeError
            return int(value)
        if base == "float":
            if isinstance(value, bool):
                raise TypeError
            return float(value)
        if base == "bool":
            if isinstance(value, bool):
                return value
            text = str(value).strip().lower()
            if text in _TRUE:
                return True
            if text in _FALSE:
                return False
            raise ValueError
    except (TypeError, ValueError):
        raise ConfigError(f"embedder.{name} must be a {base}, got {value!r}") from None
    raise AssertionError(f"unhandled kind {kind}")  # pragma: no cover


def load_config(
    path: Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> EmbedderConfig:
    """Load config from TOML plus KIBBLE_EMBED_* environment overrides."""
    environ = os.environ if env is None else env
    explicit = path is not None
    file_path = path if path is not None else (cwd or Path.cwd()) / ".kibble" / "config.toml"

    table: dict[str, Any] = {}
    if file_path.exists():
        try:
            parsed = tomllib.loads(file_path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{file_path} is not valid TOML: {exc}") from exc
        section = parsed.get("embedder", {})
        if not isinstance(section, dict):
            raise ConfigError(f"[embedder] in {file_path} must be a table")
        table = dict(section)
    elif explicit:
        raise ConfigError(f"config file not found: {file_path}")

    overrides = {
        name: environ[f"KIBBLE_EMBED_{name.upper()}"]
        for name in _KINDS
        if f"KIBBLE_EMBED_{name.upper()}" in environ
    }
    return EmbedderConfig.from_mapping({**table, **overrides})


def resolve_api_key(config: EmbedderConfig, env: Mapping[str, str] | None = None) -> str | None:
    """The API key named by `api_key_env`, or None for unauthenticated endpoints."""
    if config.api_key_env is None:
        return None
    environ = os.environ if env is None else env
    value = environ.get(config.api_key_env, "")
    if not value:
        raise ConfigError(
            f"environment variable {config.api_key_env} is not set "
            "(it is named by embedder.api_key_env)"
        )
    return value


def cleartext_key_warning(config: EmbedderConfig, has_key: bool) -> str | None:
    """Warn when an API key would travel over plain http to a non-loopback host."""
    parts = urlsplit(config.base_url)
    if not has_key or parts.scheme != "http":
        return None
    host = parts.hostname or ""
    if host == "localhost":
        return None
    try:
        if ipaddress.ip_address(host).is_loopback:
            return None
    except ValueError:
        pass
    return (
        f"the API key would be sent over plain http to {host}; use https, or an SSH "
        "tunnel, unless this network is fully trusted"
    )
