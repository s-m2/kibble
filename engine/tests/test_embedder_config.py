from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kibble_engine.embedders import (
    ConfigError,
    EmbedderConfig,
    cleartext_key_warning,
    compute_embedder_id,
    load_config,
    resolve_api_key,
)

BASE = {"base_url": "http://localhost:11434/v1/", "model": "nomic-embed-text"}


def test_defaults_and_normalisation() -> None:
    cfg = EmbedderConfig.from_mapping(BASE)
    assert cfg.base_url == "http://localhost:11434/v1"  # trailing slash stripped
    assert cfg.endpoint == "http://localhost:11434/v1/embeddings"
    assert cfg.host == "localhost:11434"
    assert (cfg.batch_size, cfg.concurrency, cfg.dimensions, cfg.api_key_env) == (64, 4, None, None)
    assert cfg.verify_tls is True


def test_string_values_are_coerced_like_environment_variables() -> None:
    cfg = EmbedderConfig.from_mapping(
        {**BASE, "batch_size": "32", "timeout_s": "2.5", "verify_tls": "false", "dimensions": ""}
    )
    assert (cfg.batch_size, cfg.timeout_s, cfg.verify_tls, cfg.dimensions) == (32, 2.5, False, None)


def test_requests_per_minute_is_optional_and_coerced() -> None:
    assert EmbedderConfig.from_mapping(BASE).requests_per_minute is None
    cfg = EmbedderConfig.from_mapping({**BASE, "requests_per_minute": "15"})
    assert cfg.requests_per_minute == 15


def test_there_is_no_default_endpoint() -> None:
    with pytest.raises(ConfigError, match=r"missing required.*base_url.*model.*by design"):
        EmbedderConfig.from_mapping({})


def test_unknown_settings_are_rejected_with_the_known_list() -> None:
    with pytest.raises(ConfigError, match=r"unknown embedder setting.*batchsize.*Known settings"):
        EmbedderConfig.from_mapping({**BASE, "batchsize": 3})


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"base_url": "localhost:11434"}, "http"),
        ({"base_url": "ftp://x/v1"}, "http"),
        ({"model": "  "}, "model"),
        ({"batch_size": 0}, "batch_size"),
        ({"concurrency": 1000}, "concurrency"),
        ({"max_retries": -1}, "max_retries"),
        ({"timeout_s": 0}, "positive"),
        ({"dimensions": 0}, "dimensions"),
        ({"batch_size": "many"}, "must be a int"),
        ({"verify_tls": "maybe"}, "must be a bool"),
        ({"ca_bundle": "/etc/ca.pem", "verify_tls": False}, "contradict"),
        ({"slice_dimensions": 0}, "slice_dimensions"),
        ({"requests_per_minute": 0}, "requests_per_minute"),
        ({"dimensions": 512, "slice_dimensions": 512}, "alternatives"),
    ],
)
def test_invalid_values_fail_with_a_pointed_message(
    override: dict[str, object], message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        EmbedderConfig.from_mapping({**BASE, **override})


# ---------------------------------------------------------------- loading


def write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_load_from_toml_with_env_overrides(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        '[embedder]\nbase_url = "http://gpu-box:8080/v1"\nmodel = "m1"\nbatch_size = 16\n',
    )
    cfg = load_config(path, env={"KIBBLE_EMBED_MODEL": "m2", "KIBBLE_EMBED_CONCURRENCY": "2"})
    assert (cfg.model, cfg.batch_size, cfg.concurrency) == ("m2", 16, 2)  # env beats file


def test_environment_alone_is_enough(tmp_path: Path) -> None:
    env = {"KIBBLE_EMBED_BASE_URL": "http://localhost:1234/v1", "KIBBLE_EMBED_MODEL": "m"}
    assert load_config(cwd=tmp_path, env=env).model == "m"


def test_default_location_is_dot_kibble_config(tmp_path: Path) -> None:
    (tmp_path / ".kibble").mkdir()
    (tmp_path / ".kibble" / "config.toml").write_text(
        '[embedder]\nbase_url = "http://localhost:1/v1"\nmodel = "from-file"\n', encoding="utf-8"
    )
    assert load_config(cwd=tmp_path, env={}).model == "from-file"


def test_missing_explicit_file_and_bad_toml_are_clear_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml", env={})
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_config(write_config(tmp_path, "[embedder\n"), env={})
    with pytest.raises(ConfigError, match="must be a table"):
        load_config(write_config(tmp_path, 'embedder = "x"\n'), env={})


# ------------------------------------------------------------- keys and warnings


def test_api_key_comes_from_the_named_environment_variable() -> None:
    plain = EmbedderConfig.from_mapping(BASE)
    keyed = EmbedderConfig.from_mapping({**BASE, "api_key_env": "MY_KEY"})
    assert resolve_api_key(plain, {}) is None
    assert resolve_api_key(keyed, {"MY_KEY": "secret"}) == "secret"
    with pytest.raises(ConfigError, match="MY_KEY is not set"):
        resolve_api_key(keyed, {})
    with pytest.raises(ConfigError, match="MY_KEY is not set"):
        resolve_api_key(keyed, {"MY_KEY": ""})


@pytest.mark.parametrize(
    ("url", "has_key", "warns"),
    [
        ("http://gpu-box:8080/v1", True, True),
        ("http://192.168.1.5:8080/v1", True, True),
        ("http://localhost:11434/v1", True, False),
        ("http://127.0.0.1:8080/v1", True, False),
        ("http://[::1]:8080/v1", True, False),
        ("https://openrouter.ai/api/v1", True, False),
        ("http://gpu-box:8080/v1", False, False),
    ],
)
def test_cleartext_key_warning(url: str, has_key: bool, warns: bool) -> None:
    cfg = EmbedderConfig.from_mapping({"base_url": url, "model": "m"})
    assert (cleartext_key_warning(cfg, has_key) is not None) is warns


# ----------------------------------------------------------------- embedder id


def test_embedder_id_tracks_only_what_changes_stored_vectors() -> None:
    base = compute_embedder_id("m", 768, "doc: ", None)
    assert base == compute_embedder_id("m", 768, "doc: ", None)  # stable
    assert base.startswith("m@768:")
    assert base != compute_embedder_id("other", 768, "doc: ", None)
    assert base != compute_embedder_id("m", 384, "doc: ", None)
    assert base != compute_embedder_id("m", 768, "", None)
    assert base != compute_embedder_id("m", 768, "doc: ", 768)
    assert base != compute_embedder_id("m", 768, "doc: ", None, 768)


def test_ids_made_before_slice_dimensions_existed_are_unchanged() -> None:
    legacy = json.dumps(
        {"model": "m", "dim": 768, "doc_prefix": "doc: ", "dimensions": None}, sort_keys=True
    )
    digest = hashlib.sha256(legacy.encode()).hexdigest()[:8]
    assert compute_embedder_id("m", 768, "doc: ", None) == f"m@768:{digest}"
    assert compute_embedder_id("m", 768, "doc: ", None, None) == f"m@768:{digest}"
