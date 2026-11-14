"""Detect when the model behind an endpoint changes under an existing index.

Hosted endpoints (OpenRouter among them) often don't say which quantization or build
they serve, and may change it without notice. Vectors from two different builds of
"the same" model are close but not identical, and mixing them in one index quietly
lowers retrieval quality. The canary is one fixed text, embedded the first time an
index is used with an embedder and re-embedded on later runs: if the vector moved, the
endpoint changed and the index should be re-embedded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, Protocol

from kibble_engine.embedders.base import cosine
from kibble_engine.embedders.openai_compatible import OpenAICompatibleEmbedder

CANARY_TEXT = "def validate_token(token):\n    return token.expires_at > now()"

# BF16 on different GPUs agrees to ~4+ nines; a different quantization or build usually
# lands below this. Tunable, but err towards noticing.
DEFAULT_THRESHOLD = 0.999


class MetaStore(Protocol):
    def get_meta(self, key: str) -> str | None: ...

    def set_meta(self, key: str, value: str) -> None: ...


@dataclass(frozen=True, slots=True)
class CanaryResult:
    status: Literal["recorded", "ok", "drift"]
    similarity: float | None  # cosine to the recorded canary; None when just recorded

    @property
    def message(self) -> str:
        if self.status == "recorded":
            return "recorded a reference vector for this embedder (future runs will compare)"
        if self.status == "ok":
            return f"endpoint output matches the recorded reference (cosine {self.similarity:.6f})"
        return (
            f"the endpoint's output has changed since this index was built (cosine "
            f"{self.similarity:.6f} to the recorded reference). It may now serve a different "
            "build or quantization of the model; re-embed the index to keep results consistent."
        )


def check_canary(
    embedder: OpenAICompatibleEmbedder,
    store: MetaStore,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> CanaryResult:
    """Embed the canary and compare it with the vector recorded for this embedder id."""
    vector = embedder.embed_documents([CANARY_TEXT])[0]
    key = f"canary:{embedder.info.embedder_id}"  # info is available once a response was seen
    recorded = store.get_meta(key)
    if recorded is None:
        store.set_meta(key, json.dumps([round(x, 7) for x in vector]))
        return CanaryResult("recorded", None)
    similarity = cosine(json.loads(recorded), vector)
    return CanaryResult("ok" if similarity >= threshold else "drift", similarity)
