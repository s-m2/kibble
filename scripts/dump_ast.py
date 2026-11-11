"""Print a file's tree-sitter syntax tree, to debug the language adapters.

    uv run python scripts/dump_ast.py path/to/file.ts            # named nodes only
    uv run python scripts/dump_ast.py path/to/file.ts --all      # include punctuation tokens

Each line is `field: node_type [start_byte-end_byte] "text preview"`. If the chunker
mislabels something, compare the node types here with parsing/languages.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from kibble_engine.parsing import chunker
from kibble_engine.parsing.languages import language_for_path


def dump(node: Any, source: bytes, depth: int, field: str | None, show_all: bool) -> None:
    if not show_all and not node.is_named:
        return
    text = source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
    preview = " ".join(text.split())[:48]
    label = f"{field}: " if field else ""
    print(f"{'  ' * depth}{label}{node.type} [{node.start_byte}-{node.end_byte}] {preview!r}")
    for i, child in enumerate(node.children):
        dump(child, source, depth + 1, node.field_name_for_child(i), show_all)


def main(argv: list[str]) -> int:
    if not argv or argv[0].startswith("-"):
        print(__doc__)
        return 2
    path = Path(argv[0])
    spec = language_for_path(path.name)
    if spec is None:
        print(f"no language adapter for {path.suffix!r}")
        return 1
    data = path.read_bytes()
    dump(chunker._parse(spec, data), data, 0, None, "--all" in argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
