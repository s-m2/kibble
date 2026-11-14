"""Embeddings client for any server speaking the OpenAI embeddings wire format.

Hosted providers (OpenRouter, OpenAI, ...), local servers (Ollama, LM Studio,
llama.cpp, vLLM) and LAN gateways all share one code path. What differs between
them is handled here instead of leaking into the rest of the engine:

- batch limits: a rejected batch is halved and retried, recursively
- input length limits: over-long text is truncated up front, and a single input
  the server still rejects is shrunk and retried a few times
- transient failures (connection errors, timeouts, 429, 5xx): exponential backoff,
  honouring Retry-After
- hard failures (auth, credits, wrong URL/model): raised immediately with a message
  that says what to check
- result order: restored from each item's `index`, never assumed
- output size: learned from the first response and enforced from then on

The client is synchronous and thread-safe; `embed_documents` runs batches on a
small thread pool.
"""

from __future__ import annotations

import logging
import math
import random
import ssl
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import Any

from kibble_engine.embedders.base import (
    EmbedderInfo,
    EmbedderStats,
    ProbeResult,
    ProgressCallback,
    compute_embedder_id,
)
from kibble_engine.embedders.config import EmbedderConfig, cleartext_key_warning
from kibble_engine.embedders.errors import (
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
from kibble_engine.embedders.http import httpx

log = logging.getLogger(__name__)

# 408/524: timeouts; 529: "provider overloaded" (all documented by OpenRouter).
_RETRYABLE = frozenset({408, 429, 500, 502, 503, 504, 524, 529})
_INPUT_REJECTED = frozenset({400, 413, 422})
_MAX_BACKOFF_S = 30.0
_MAX_RETRY_AFTER_S = 60.0
_MIN_SHRINKABLE_CHARS = 64
_SHRINK_ATTEMPTS = 3
_PROBE_TEXT = "kibble probe"


class _InputRejected(Exception):
    """Internal: the server refused this request's input (size, length, format)."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"{status}: {message}")
        self.status = status
        self.message = message


class _Pacer:
    """Spaces requests evenly so a per-minute quota is never exceeded."""

    def __init__(
        self, per_minute: int, clock: Callable[[], float], sleep: Callable[[float], None]
    ) -> None:
        self._interval = 60.0 / per_minute
        self._clock = clock
        self._sleep = sleep
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            start = max(now, self._next)
            self._next = start + self._interval
        if start > now:
            self._sleep(start - now)


class OpenAICompatibleEmbedder:
    def __init__(
        self,
        config: EmbedderConfig,
        *,
        api_key: str | None = None,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._api_key = api_key
        self._sleep = sleep
        self._clock = clock
        self._pacer = (
            _Pacer(config.requests_per_minute, clock, sleep) if config.requests_per_minute else None
        )
        self._lock = threading.Lock()
        self._dim: int | None = None
        self.stats = EmbedderStats()
        self._owns_client = client is None
        self._client = client if client is not None else self._build_client()

        warning = cleartext_key_warning(config, api_key is not None)
        if warning:
            log.warning("embedder: %s", warning)

    # ------------------------------------------------------------- lifecycle

    def _build_client(self) -> Any:
        cfg = self._config
        verify: Any = True
        if cfg.ca_bundle:
            verify = ssl.create_default_context(cafile=cfg.ca_bundle)
        elif not cfg.verify_tls:
            log.warning("embedder: TLS verification is DISABLED for %s", cfg.host)
            verify = False
        return httpx.Client(timeout=self._timeout(cfg.timeout_s), verify=verify)

    @staticmethod
    def _timeout(seconds: float) -> Any:
        return httpx.Timeout(seconds, connect=min(10.0, seconds))

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OpenAICompatibleEmbedder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def with_overrides(self, **changes: Any) -> OpenAICompatibleEmbedder:
        """A sibling embedder with some settings changed (e.g. prefixes), sharing this
        one's HTTP client, key and request pacing, so quota accounting stays correct."""
        clone = OpenAICompatibleEmbedder(
            replace(self._config, **changes),
            api_key=self._api_key,
            client=self._client,
            sleep=self._sleep,
            clock=self._clock,
        )
        clone._pacer = self._pacer
        return clone

    # ---------------------------------------------------------------- public

    @property
    def info(self) -> EmbedderInfo:
        if self._dim is None:
            raise EmbedderError("embedding dimension is not known yet; call probe() first")
        cfg = self._config
        return EmbedderInfo(
            compute_embedder_id(
                cfg.model, self._dim, cfg.doc_prefix, cfg.dimensions, cfg.slice_dimensions
            ),
            cfg.model,
            self._dim,
            cfg.host,
        )

    def probe(self) -> ProbeResult:
        """One real request with the long first-request timeout; learns the dimension."""
        started = time.monotonic()
        try:
            vectors = self._request(
                [self._prepare(_PROBE_TEXT, self._config.doc_prefix)],
                timeout=self._config.probe_timeout_s,
            )
        except _InputRejected as rejected:
            raise EmbedderRequestError(
                f"{self._config.endpoint} rejected even a tiny test input: {rejected.message} "
                f"(HTTP {rejected.status}). Check the model name and the `dimensions` setting."
            ) from rejected
        assert self._dim is not None and len(vectors) == 1
        return ProbeResult(self._dim, time.monotonic() - started)

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([self._prepare(text, self._config.query_prefix)])[0]

    def embed_documents(
        self, texts: Sequence[str], *, on_progress: ProgressCallback | None = None
    ) -> list[list[float]]:
        return self._embed_many(texts, self._config.doc_prefix, on_progress)

    def embed_queries(
        self, texts: Sequence[str], *, on_progress: ProgressCallback | None = None
    ) -> list[list[float]]:
        return self._embed_many(texts, self._config.query_prefix, on_progress)

    def _embed_many(
        self, texts: Sequence[str], prefix: str, on_progress: ProgressCallback | None
    ) -> list[list[float]]:
        prepared = [self._prepare(t, prefix) for t in texts]
        total = len(prepared)
        if total == 0:
            return []
        size = self._config.batch_size
        starts = list(range(0, total, size))
        results: list[list[float]] = [[] for _ in range(total)]
        done = 0

        def finish(start: int, vectors: list[list[float]]) -> None:
            nonlocal done
            results[start : start + len(vectors)] = vectors
            done += len(vectors)
            if on_progress is not None:
                on_progress(done, total)

        workers = min(self._config.concurrency, len(starts))
        if workers <= 1:
            for start in starts:
                finish(start, self._embed_batch(prepared[start : start + size]))
            return results

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures: dict[Future[list[list[float]]], int] = {
                pool.submit(self._embed_batch, prepared[s : s + size]): s for s in starts
            }
            try:
                for future in as_completed(futures):
                    finish(futures[future], future.result())
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        return results

    # -------------------------------------------------------------- internals

    def _prepare(self, text: str, prefix: str) -> str:
        """Apply the prefix and the length cap. The prefix always survives truncation."""
        if not text.strip():
            raise ValueError("cannot embed empty text")
        limit = self._config.max_input_chars
        if len(prefix) + len(text) > limit:
            self.stats.bump("truncations")
            text = text[: max(1, limit - len(prefix))]
        return prefix + text

    def _embed_batch(self, inputs: list[str]) -> list[list[float]]:
        """Embed `inputs`, adapting to server-side batch and length limits."""
        try:
            return self._request(inputs)
        except _InputRejected as rejected:
            if len(inputs) > 1:
                self.stats.bump("batch_splits")
                log.info("embedder: batch of %d rejected (%s); splitting", len(inputs), rejected)
                middle = len(inputs) // 2
                return self._embed_batch(inputs[:middle]) + self._embed_batch(inputs[middle:])
            return self._embed_single_shrinking(inputs[0], rejected)

    def _embed_single_shrinking(self, text: str, first: _InputRejected) -> list[list[float]]:
        """A lone input was rejected: assume it's too long and retry with less of it."""
        last = first
        for _ in range(_SHRINK_ATTEMPTS):
            if len(text) <= _MIN_SHRINKABLE_CHARS:
                break
            text = text[: len(text) // 2]
            self.stats.bump("input_shrinks")
            log.warning("embedder: input rejected (%s); retrying with %d chars", last, len(text))
            try:
                return self._request([text])
            except _InputRejected as rejected:
                last = rejected
        raise EmbedderRequestError(
            f"{self._config.endpoint} rejected the request: {last.message} (HTTP {last.status}). "
            "If this is a length limit, lower embedder.max_input_chars and the chunker's "
            "max_chars; otherwise check the model name and the `dimensions` setting."
        )

    def _request(self, inputs: list[str], *, timeout: float | None = None) -> list[list[float]]:
        cfg = self._config
        body: dict[str, Any] = {"model": cfg.model, "input": inputs, "encoding_format": "float"}
        if cfg.dimensions is not None:
            body["dimensions"] = cfg.dimensions
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        seconds = timeout if timeout is not None else cfg.timeout_s

        for attempt in range(cfg.max_retries + 1):
            last_attempt = attempt == cfg.max_retries
            if self._pacer is not None:
                self._pacer.wait()
            self.stats.bump("requests")
            try:
                response = self._client.post(
                    cfg.endpoint, json=body, headers=headers, timeout=self._timeout(seconds)
                )
            except httpx.TransportError as exc:
                if last_attempt:
                    raise EmbedderConnectionError(self._describe_transport(exc, seconds)) from exc
                self._backoff(attempt, None)
                continue

            status = int(response.status_code)
            if status == 200:
                return self._parse(response, len(inputs))

            message = _error_message(response)
            if status in _RETRYABLE:
                if last_attempt:
                    where = f"{cfg.endpoint} kept returning HTTP {status}: {message}"
                    if status == 429:
                        raise EmbedderRateLimitError(
                            f"{where}. You are being rate limited: set "
                            "embedder.requests_per_minute and concurrency=1, or wait. Hosted "
                            "free tiers usually also have a daily request cap, which only "
                            "waiting (or a paid plan) resolves."
                        )
                    raise EmbedderServerError(where)
                self._backoff(attempt, _retry_after(response))
                continue
            raise self._terminal_error(status, message)
        raise AssertionError("unreachable")  # pragma: no cover

    def _terminal_error(self, status: int, message: str) -> Exception:
        """Map a non-retryable status to an exception (or the internal rejection signal)."""
        cfg = self._config
        where = f"{cfg.endpoint} returned HTTP {status}: {message}"
        if status in (401, 403):
            hint = (
                f"check the key in ${cfg.api_key_env}"
                if cfg.api_key_env
                else "this endpoint needs a key: set embedder.api_key_env"
            )
            return EmbedderAuthError(f"{where}. {hint}")
        if status == 402:
            return EmbedderCreditsError(f"{where}. The provider reports insufficient credits.")
        if status == 404:
            return EmbedderNotFoundError(
                f"{where}. Check embedder.base_url (many servers need a /v1 suffix) and that "
                f"model {cfg.model!r} exists on this server."
            )
        if status in _INPUT_REJECTED:
            return _InputRejected(status, message)
        return EmbedderRequestError(where)

    def _backoff(self, attempt: int, retry_after: float | None) -> None:
        self.stats.bump("retries")
        if retry_after is not None:
            delay = min(retry_after, _MAX_RETRY_AFTER_S)
        else:
            delay = min(self._config.backoff_base_s * 2**attempt, _MAX_BACKOFF_S)
            delay *= 1.0 + 0.25 * random.random()  # jitter, so parallel workers spread out
        log.info("embedder: transient failure; retrying in %.1fs", delay)
        self._sleep(delay)

    def _describe_transport(self, exc: Exception, seconds: float) -> str:
        endpoint = self._config.endpoint
        if isinstance(exc, httpx.TimeoutException):
            return (
                f"request to {endpoint} timed out after {seconds:g}s. A local server may need "
                "longer while it loads the model: raise embedder.probe_timeout_s / timeout_s."
            )
        return f"could not connect to {endpoint}: {exc}. Is the server running and reachable?"

    def _parse(self, response: Any, expected: int) -> list[list[float]]:
        endpoint = self._config.endpoint
        try:
            payload = response.json()
        except ValueError as exc:
            raise EmbedderResponseError(f"{endpoint} returned a non-JSON body") from exc
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or len(data) != expected:
            got = len(data) if isinstance(data, list) else "no"
            raise EmbedderResponseError(
                f"{endpoint} returned {got} embeddings for {expected} inputs"
            )
        if all(isinstance(item, dict) and isinstance(item.get("index"), int) for item in data):
            data = sorted(data, key=lambda item: item["index"])
            if [item["index"] for item in data] != list(range(expected)):
                raise EmbedderResponseError(f"{endpoint} returned unexpected item indexes")

        vectors: list[list[float]] = []
        for item in data:
            raw = item.get("embedding") if isinstance(item, dict) else None
            if (
                not isinstance(raw, list)
                or not raw
                or not all(isinstance(x, (int, float)) for x in raw)
            ):
                raise EmbedderResponseError(
                    f"{endpoint} returned an embedding that is not a list of numbers "
                    "(does the server support encoding_format='float'?)"
                )
            vectors.append([float(x) for x in raw])
        if self._config.slice_dimensions is not None:
            vectors = [self._slice(vector) for vector in vectors]
        self._check_dimension(vectors)
        return vectors

    def _slice(self, vector: list[float]) -> list[float]:
        """Matryoshka-style truncation: keep the first N dimensions, re-normalise to unit length."""
        keep = self._config.slice_dimensions
        assert keep is not None
        if len(vector) < keep:
            raise EmbedderDimensionError(
                f"the model returned {len(vector)}-dimensional vectors, so they can't be "
                f"sliced to embedder.slice_dimensions={keep}"
            )
        head = vector[:keep]
        norm = math.sqrt(sum(x * x for x in head))
        if norm == 0.0:
            raise EmbedderResponseError("a sliced embedding had zero length")
        return [x / norm for x in head]

    def _check_dimension(self, vectors: list[list[float]]) -> None:
        wanted = self._config.dimensions
        with self._lock:
            if self._dim is None:
                self._dim = len(vectors[0])
                if wanted is not None and self._dim != wanted:
                    self._dim = None
                    raise EmbedderDimensionError(
                        f"embedder.dimensions={wanted} was requested but the server returned "
                        f"{len(vectors[0])}-dimensional vectors; this endpoint or model "
                        "probably ignores the `dimensions` parameter. Remove it; if the model "
                        "supports Matryoshka-style truncation, use embedder.slice_dimensions "
                        "instead (the client keeps the first N dimensions and re-normalises)."
                    )
            for vector in vectors:
                if len(vector) != self._dim:
                    raise EmbedderDimensionError(
                        f"server returned a {len(vector)}-dimensional vector but this index "
                        f"uses {self._dim}; the model behind the endpoint may have changed"
                    )


def _error_message(response: Any) -> str:
    """Best-effort human-readable message from an error response body."""
    try:
        payload = response.json()
    except ValueError:
        return str(response.text)[:200].strip() or "no message"
    if isinstance(payload, dict):
        error = payload.get("error", payload.get("detail", payload.get("message")))
        if isinstance(error, dict):
            error = error.get("message") or error
        if error:
            return str(error)[:300]
    return str(payload)[:200]


def _retry_after(response: Any) -> float | None:
    raw = response.headers.get("retry-after")
    try:
        return max(0.0, float(raw)) if raw is not None else None
    except ValueError:
        return None  # HTTP-date form: fall back to our own backoff
