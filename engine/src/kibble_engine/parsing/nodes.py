"""The minimal syntax-node surface the chunker relies on.

tree_sitter.Node satisfies this structurally. Keeping the chunker on this small
protocol (instead of tree-sitter queries) means it survives binding API changes
and can be unit-tested with hand-built trees.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class SyntaxNode(Protocol):
    @property
    def type(self) -> str: ...

    @property
    def start_byte(self) -> int: ...

    @property
    def end_byte(self) -> int: ...

    @property
    def children(self) -> Sequence[SyntaxNode]: ...

    @property
    def has_error(self) -> bool: ...

    def child_by_field_name(self, name: str, /) -> SyntaxNode | None: ...
