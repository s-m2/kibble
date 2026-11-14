"""An in-process fake of an OpenAI-compatible /embeddings server.

It plugs into the embedder through an httpx mock transport, so tests need no network
and no ports. Vectors are deterministic and *bag-of-words based*: texts that share
words get positive cosine similarity, which makes retrieval-plumbing tests
meaningful without a real model. Behaviours real servers disagree on can be switched
on: batch and length limits, auth, shuffled result order, ignoring `dimensions`, and
injected failures (HTTP statuses, dropped connections).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import threading
from collections import deque
from collections.abc import Callable
from typing import Any

from kibble_engine.embedders.http import httpx

Handler = Callable[[Any], Any]


class FakeEmbeddingServer:
    def __init__(
        self,
        *,
        dim: int = 16,
        model: str = "fake-embed",
        api_key: str | None = None,
        max_batch: int | None = None,
        max_input_chars: int | None = None,
        path: str = "/v1/embeddings",
        honor_dimensions: bool = True,
    ) -> None:
        self.dim = dim
        self.model = model
        self.api_key = api_key
        self.max_batch = max_batch
        self.max_input_chars = max_input_chars
        self.path = path
        self.honor_dimensions = honor_dimensions
        self.shuffle = False  # return items in reverse order (clients must use `index`)
        self.reject_all_with: int | None = None  # e.g. 400 for an unsupported parameter
        self.wrong_count = False  # return one embedding too few
        self.perturb = 0.0  # add deterministic noise: simulates a different build/quantization
        self.jitter = 0.0  # noise that differs per request: a non-deterministic backend
        self.prefix_breaks_model = False  # a prefixed text loses all word overlap
        self.requests: list[dict[str, Any]] = []  # every request body, accepted or not
        self.accepted_batch_sizes: list[int] = []
        self.auth_headers: list[str | None] = []
        self._script: deque[Handler] = deque()
        self._lock = threading.Lock()
        self.transport = httpx.MockTransport(self._handle)

    # ----------------------------------------------------------- test controls

    def client(self) -> Any:
        return httpx.Client(transport=self.transport)

    def fail_next(
        self,
        status: int,
        *,
        times: int = 1,
        headers: dict[str, str] | None = None,
        message: str = "injected failure",
    ) -> None:
        for _ in range(times):
            self._script.append(
                lambda request, s=status, h=headers, m=message: httpx.Response(
                    s, json={"error": {"message": m}}, headers=h or {}
                )
            )

    def drop_next(self, times: int = 1) -> None:
        def drop(request: Any) -> Any:
            raise httpx.ConnectError("injected connection failure", request=request)

        self._script.extend([drop] * times)

    def respond_next(self, response: Any) -> None:
        self._script.append(lambda request: response)

    # ------------------------------------------------------------- the server

    def _handle(self, request: Any) -> Any:
        with self._lock:
            scripted = self._script.popleft() if self._script else None
        if scripted is not None:
            return scripted(request)

        if request.url.path != self.path:
            return _error(404, "no such endpoint")
        auth = request.headers.get("authorization")
        with self._lock:
            self.auth_headers.append(auth)
        if self.api_key is not None and auth != f"Bearer {self.api_key}":
            return _error(401, "invalid api key")

        body = json.loads(request.content)
        with self._lock:
            self.requests.append(body)
            call = len(self.requests)
        if body.get("model") != self.model:
            return _error(404, f"model {body.get('model')!r} not found")
        inputs = body["input"]
        if self.reject_all_with is not None:
            return _error(self.reject_all_with, "unsupported request")
        if self.max_batch is not None and len(inputs) > self.max_batch:
            return _error(400, f"batch too large (max {self.max_batch})")
        if self.max_input_chars is not None and any(len(t) > self.max_input_chars for t in inputs):
            return _error(400, f"input too long (max {self.max_input_chars} characters)")

        dim = body.get("dimensions") if self.honor_dimensions else None
        dim = int(dim or self.dim)
        items = [
            {"object": "embedding", "index": i, "embedding": self._vector(text, dim, call)}
            for i, text in enumerate(inputs)
        ]
        if self.wrong_count:
            items = items[:-1]
        if self.shuffle:
            items.reverse()
        with self._lock:
            self.accepted_batch_sizes.append(len(inputs))
        return httpx.Response(200, json={"object": "list", "data": items, "model": self.model})

    _PREFIXES = ("query: ", "passage: ", "search_query: ", "search_document: ")

    def _vector(self, text: str, dim: int, call: int = 0) -> list[float]:
        if self.prefix_breaks_model and text.startswith(self._PREFIXES):
            digest = hashlib.sha256(text.encode()).hexdigest()  # one opaque token: no overlap
            vector = self.embed_text(digest, dim)
        else:
            vector = self.embed_text(text, dim)
        if self.perturb:
            rng = random.Random(f"kibble-noise:{text}")
            vector = [v + self.perturb * rng.uniform(-1.0, 1.0) for v in vector]
        if self.jitter:
            rng = random.Random(f"kibble-jitter:{call}:{text}")
            vector = [v + self.jitter * rng.uniform(-1.0, 1.0) for v in vector]
        return vector

    @staticmethod
    def embed_text(text: str, dim: int) -> list[float]:
        """Deterministic bag-of-words vector: shared words => positive cosine similarity."""
        tokens = re.findall(r"[a-z0-9]+", text.lower()) or [text]
        vector = [0.0] * dim
        for token in tokens:
            rng = random.Random(f"kibble-fake:{token}")
            for i in range(dim):
                vector[i] += rng.uniform(-1.0, 1.0)
        return vector


def _error(status: int, message: str) -> Any:
    return httpx.Response(status, json={"error": {"message": message}})
