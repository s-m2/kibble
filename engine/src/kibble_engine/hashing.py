"""Content hashing helpers shared by the chunker, the store and the tests."""

from __future__ import annotations

import hashlib


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
