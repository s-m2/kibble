"""Embedder errors. Each one names a specific, fixable problem; none are generic."""

from __future__ import annotations


class EmbedderError(Exception):
    """Base class: anything that went wrong configuring or calling an embedder."""


class ConfigError(EmbedderError):
    """The embedder configuration is missing, malformed, or inconsistent."""


class EmbedderConnectionError(EmbedderError):
    """Could not reach the endpoint (refused, DNS, TLS, timeout) after retries."""


class EmbedderAuthError(EmbedderError):
    """401/403: the key is missing, wrong, or not allowed to use this model."""


class EmbedderCreditsError(EmbedderError):
    """402: the provider reports no remaining credits."""


class EmbedderNotFoundError(EmbedderError):
    """404: wrong base URL path (often a missing /v1) or unknown model."""


class EmbedderRateLimitError(EmbedderError):
    """429 persisted after retries."""


class EmbedderServerError(EmbedderError):
    """5xx persisted after retries."""


class EmbedderRequestError(EmbedderError):
    """The server rejected the request itself (400/413/422...) and we couldn't adapt."""


class EmbedderResponseError(EmbedderError):
    """The server answered 200 but the body wasn't a valid embeddings response."""


class EmbedderDimensionError(EmbedderError):
    """Vector length changed or doesn't match the configured `dimensions`."""
