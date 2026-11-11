"""Identifier splitting, so BM25 can match `token` against `validateToken`."""

from __future__ import annotations

import re

_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|[_\-\s.$]+")


def split_identifier(name: str) -> list[str]:
    """Split on case changes and separators: `parseConfigV2` -> parse, config, v2."""
    return [p.lower() for p in _BOUNDARY.split(name) if p]


def symbol_terms(*names: str | None) -> str:
    """Space-joined original names plus their parts, deduplicated, order preserved.

    This is the text for the FTS `symbol_terms` column: the exact identifier still
    matches exactly, and its words match natural-language queries.
    """
    seen: dict[str, None] = {}
    for name in names:
        if not name:
            continue
        seen.setdefault(name)
        for part in split_identifier(name):
            seen.setdefault(part)
    return " ".join(seen)
