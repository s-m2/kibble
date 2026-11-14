"""`kibble-engine doctor`: check an embedder setup before indexing anything.

Each check returns ok / warn / fail with a sentence that says what to do next.
A failed endpoint check stops the rest, since nothing else would be meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from kibble_engine.embedders.base import cosine
from kibble_engine.embedders.canary import DEFAULT_THRESHOLD
from kibble_engine.embedders.config import EmbedderConfig, cleartext_key_warning
from kibble_engine.embedders.errors import EmbedderError
from kibble_engine.embedders.openai_compatible import OpenAICompatibleEmbedder

Status = Literal["ok", "warn", "fail"]

# Model families whose cards recommend asymmetric prefixes. Checked by substring.
_PREFIX_HINTS: tuple[tuple[str, str], ...] = (
    ("e5", 'E5 models expect "passage: " for documents and "query: " for queries'),
    (
        "nemotron-3-embed",
        'Nemotron 3 Embed expects "passage: " for documents and "query: " for queries',
    ),
    ("nomic-embed", 'nomic-embed expects "search_document: " and "search_query: "'),
    ("bge", "BGE models often want a query instruction; see the model card"),
)


# A smoke test, not a benchmark: each query should rank its own snippet first. It exists to
# catch grossly wrong setups (wrong model behind the endpoint, missing or doubled prefixes).
_PAIRS: tuple[tuple[str, str], ...] = (
    (
        "count how often each element occurs in a list",
        "def count_each(items):\n    result = {}\n    for item in items:\n"
        "        result[item] = result.get(item, 0) + 1\n    return result",
    ),
    (
        "check whether a login token has expired",
        "function isExpired(token) {\n  const payload = decode(token);\n"
        "  return payload.exp * 1000 < Date.now();\n}",
    ),
    (
        "load settings from a json file",
        "def load_settings(path):\n    with open(path) as f:\n        return json.load(f)",
    ),
    (
        "retry the request with exponential backoff",
        "async function fetchJson(url, retries = 3) {\n"
        "  // retry the request with exponential backoff\n"
        "  for (let i = 0; i < retries; i++) {\n    try { return await fetch(url); }\n"
        "    catch (e) { await sleep(2 ** i * 100); }\n  }\n}",
    ),
)
_STABILITY_TEXT = "def validate_token(token):\n    return token.expires_at > now()"


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    status: Status
    detail: str


def prefix_hint(config: EmbedderConfig) -> str | None:
    if config.doc_prefix or config.query_prefix:
        return None
    model = config.model.lower()
    for needle, hint in _PREFIX_HINTS:
        if needle in model:
            return f"{hint}. Without them retrieval quality drops silently."
    return None


def run_doctor(
    embedder: OpenAICompatibleEmbedder, config: EmbedderConfig, *, has_key: bool
) -> list[Check]:
    checks: list[Check] = []

    try:
        probe = embedder.probe()
    except EmbedderError as exc:
        return [Check("endpoint", "fail", str(exc))]
    checks.append(
        Check(
            "endpoint",
            "ok",
            f"reachable; {config.model} returns {probe.dim}-dim vectors "
            f"({probe.latency_s:.2f}s for the first request)",
        )
    )

    stats = embedder.stats
    splits_before = stats.batch_splits
    try:
        embedder.embed_documents([f"doctor batch item {i}" for i in range(config.batch_size)])
    except EmbedderError as exc:
        checks.append(Check("batching", "fail", str(exc)))
    else:
        if stats.batch_splits > splits_before:
            checks.append(
                Check(
                    "batching",
                    "warn",
                    f"the server rejected a batch of {config.batch_size}; requests are split "
                    "automatically, but lower embedder.batch_size to avoid wasted calls",
                )
            )
        else:
            checks.append(Check("batching", "ok", f"batch of {config.batch_size} accepted"))

    shrinks_before = stats.input_shrinks
    filler = " ".join(f"word{i}" for i in range(config.max_input_chars))
    long_text = filler[: config.max_input_chars]
    try:
        embedder.embed_documents([long_text])
    except EmbedderError as exc:
        checks.append(Check("input length", "fail", str(exc)))
    else:
        if stats.input_shrinks > shrinks_before:
            checks.append(
                Check(
                    "input length",
                    "warn",
                    f"the server rejected a {config.max_input_chars}-char input; lower "
                    "embedder.max_input_chars (and the chunker's max_chars) to match",
                )
            )
        else:
            checks.append(
                Check("input length", "ok", f"{config.max_input_chars}-char input accepted")
            )

    checks.append(_stability_check(embedder))
    checks.append(_retrieval_check(embedder, config))

    hint = prefix_hint(config)
    if hint:
        checks.append(Check("prefixes", "warn", hint))

    warning = cleartext_key_warning(config, has_key)
    if warning:
        checks.append(Check("transport", "warn", warning))
    elif not config.verify_tls:
        checks.append(Check("transport", "warn", "TLS verification is disabled"))

    return checks


def _stability_check(embedder: OpenAICompatibleEmbedder) -> Check:
    """The same text, alone and inside a batch, should give (nearly) the same vector."""
    try:
        alone = embedder.embed_documents([_STABILITY_TEXT])[0]
        batched = embedder.embed_documents([_STABILITY_TEXT, "an unrelated second text"])[0]
    except EmbedderError as exc:
        return Check("stability", "fail", str(exc))
    similarity = cosine(alone, batched)
    if similarity >= DEFAULT_THRESHOLD:
        return Check("stability", "ok", f"repeat calls agree (cosine {similarity:.6f})")
    return Check(
        "stability",
        "warn",
        f"the same text gave different vectors (cosine {similarity:.6f}). The endpoint may be "
        "non-deterministic, or load-balanced across different builds or quantizations; mixed "
        "vectors lower retrieval quality.",
    )


def _retrieval_accuracy(embedder: OpenAICompatibleEmbedder) -> int:
    queries = embedder.embed_queries([q for q, _ in _PAIRS])
    snippets = embedder.embed_documents([d for _, d in _PAIRS])
    correct = 0
    for i, query in enumerate(queries):
        scores = [cosine(query, snippet) for snippet in snippets]
        correct += scores.index(max(scores)) == i
    return correct


def _retrieval_check(embedder: OpenAICompatibleEmbedder, config: EmbedderConfig) -> Check:
    total = len(_PAIRS)
    try:
        configured = _retrieval_accuracy(embedder)
        if configured == total:
            return Check("retrieval", "ok", f"{total}/{total} sanity queries found their snippet")
        if config.doc_prefix or config.query_prefix:
            bare = _retrieval_accuracy(embedder.with_overrides(doc_prefix="", query_prefix=""))
            if bare > configured:
                return Check(
                    "retrieval",
                    "warn",
                    f"{configured}/{total} sanity queries succeed with your prefixes but "
                    f"{bare}/{total} without them. The endpoint may already add prefixes (or "
                    "the model doesn't use them): try removing doc_prefix and query_prefix.",
                )
    except EmbedderError as exc:
        return Check("retrieval", "fail", str(exc))
    return Check(
        "retrieval",
        "warn",
        f"only {configured}/{total} sanity queries found their snippet. Check the model name, "
        "the prefixes, and that the endpoint serves the model you expect.",
    )
