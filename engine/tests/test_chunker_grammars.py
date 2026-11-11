"""Chunking against the real tree-sitter grammars.

test_chunker_core.py covers the logic with fake trees; this file checks that the
language adapters' assumptions about node types match the actual grammars. If one of
these fails after a grammar upgrade, run `scripts/dump_ast.py` on the snippet and fix
the node type in parsing/languages.py.

These imports are deliberately NOT guarded with importorskip: a missing grammar
dependency should fail loudly, not skip silently.
"""

from __future__ import annotations

import re

import tree_sitter  # noqa: F401
import tree_sitter_javascript  # noqa: F401
import tree_sitter_python  # noqa: F401
import tree_sitter_typescript  # noqa: F401

from kibble_engine.parsing import chunk_file, language_for_path
from kibble_engine.parsing.models import ParsedFile

TS_SAMPLE = """import { Token } from "./token";

/** Validates a token. */
export function validateToken(token: Token): boolean {
  return token.expiresAt > Date.now();
}

export const refresh = async (t: Token) => {
  return { ...t, expiresAt: Date.now() + 1000 };
};

export interface Session { id: string; token: Token }

export type Role = "admin" | "user";

export enum Level { Low, High }

export class SessionManager {
  private sessions = new Map<string, Session>();

  create(id: string): Session {
    const s = { id, token: { expiresAt: 0 } };
    this.sessions.set(id, s);
    return s;
  }

  get(id: string) {
    return this.sessions.get(id);
  }
}
"""

JS_SAMPLE = """export default function handler(req, res) {
  res.end("ok");
}

class Counter {
  count = 0;
  inc = () => { this.count += 1; };
  reset() { this.count = 0; }
}
"""

PY_SAMPLE = '''import os

MAX_RETRIES = 3


@decorator
def fetch(url):
    """Fetch a URL."""
    return url


class Client:
    def __init__(self, base):
        self.base = base

    async def get(self, path):
        return self.base + path
'''

TSX_SAMPLE = """export const App = () => <div className="app">hello</div>;
"""


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def symbols(parsed: ParsedFile) -> set[tuple[str, str, str | None]]:
    return {(s.name, s.kind, s.parent) for s in parsed.symbols}


def assert_covers(parsed: ParsedFile, source: str) -> None:
    assert squash("".join(c.content for c in parsed.chunks)) == squash(source)


def test_extensions_map_to_adapters() -> None:
    names = {
        ext: getattr(language_for_path(f"x{ext}"), "name", None)
        for ext in (".ts", ".tsx", ".js", ".jsx", ".mjs", ".py", ".md")
    }
    assert names == {
        ".ts": "typescript",
        ".tsx": "tsx",
        ".js": "javascript",
        ".jsx": "javascript",
        ".mjs": "javascript",
        ".py": "python",
        ".md": None,
    }


# ------------------------------------------------------------------ TypeScript


def test_typescript_symbols() -> None:
    parsed = chunk_file("src/session.ts", TS_SAMPLE, min_chars=1)
    assert not parsed.used_fallback and not parsed.has_syntax_errors
    assert symbols(parsed) == {
        ("validateToken", "function", None),
        ("refresh", "function", None),
        ("Session", "interface", None),
        ("Role", "type", None),
        ("Level", "enum", None),
        ("SessionManager", "class", None),
        ("create", "method", "SessionManager"),
        ("get", "method", "SessionManager"),
    }
    assert_covers(parsed, TS_SAMPLE)


def test_typescript_chunk_content_and_context() -> None:
    parsed = chunk_file("src/session.ts", TS_SAMPLE, min_chars=1)
    by_symbol = {c.symbol: c for c in parsed.chunks if c.symbol}
    fn = by_symbol["validateToken"]
    assert fn.kind == "function"
    assert fn.content.startswith("/** Validates a token. */")  # JSDoc stays with its function
    assert "export function validateToken" in fn.content  # `export` stays with it too
    assert "File: src/session.ts" in fn.embed_text and "Symbol: validateToken" in fn.embed_text
    assert {"validateToken", "validate", "token"} <= set(fn.symbol_terms.split())
    imports = next(c for c in parsed.chunks if "import { Token }" in c.content)
    assert imports.kind == "statements"


def test_typescript_large_class_is_split_by_member() -> None:
    parsed = chunk_file("src/session.ts", TS_SAMPLE, max_chars=150, min_chars=1)
    create = next(c for c in parsed.chunks if c.symbol == "create")
    assert create.kind == "method" and create.parent_symbol == "SessionManager"
    assert "this.sessions.set" in create.content
    header = next(c for c in parsed.chunks if c.symbol == "SessionManager")
    assert header.content.startswith("export class SessionManager")
    assert "create(" not in header.content
    assert_covers(parsed, TS_SAMPLE)


def test_typescript_default_sizes_merge_small_definitions() -> None:
    parsed = chunk_file("src/session.ts", TS_SAMPLE)
    assert len(parsed.chunks) < len({s.name for s in parsed.symbols})  # fewer chunks than symbols
    assert {("validateToken", "function", None), ("create", "method", "SessionManager")} <= symbols(
        parsed
    )
    assert_covers(parsed, TS_SAMPLE)


def test_typescript_oversized_function_splits_into_parts() -> None:
    body = "\n".join(f"  step{i}();" for i in range(30))
    source = f"export function big() {{\n{body}\n}}\n"
    parsed = chunk_file("big.ts", source, max_chars=120, min_chars=1)
    assert len(parsed.chunks) > 1
    assert all(c.symbol == "big" and c.kind == "function" for c in parsed.chunks)
    assert [s.name for s in parsed.symbols] == ["big"]
    assert all(len(c.content) <= 130 for c in parsed.chunks)
    assert_covers(parsed, source)


def test_typescript_syntax_errors_are_tolerated() -> None:
    source = "export function ok() { return 1; }\nfunction broken( {\n"
    parsed = chunk_file("bad.ts", source)
    assert parsed.has_syntax_errors and not parsed.used_fallback
    assert ("ok", "function", None) in symbols(parsed)
    assert_covers(parsed, source)


# ------------------------------------------------------------------ JS / TSX / Python


def test_javascript_symbols() -> None:
    parsed = chunk_file("src/app.js", JS_SAMPLE, min_chars=1)
    assert not parsed.used_fallback
    assert symbols(parsed) == {
        ("handler", "function", None),
        ("Counter", "class", None),
        ("inc", "method", "Counter"),
        ("reset", "method", "Counter"),
    }
    assert_covers(parsed, JS_SAMPLE)


def test_tsx_parses_jsx() -> None:
    parsed = chunk_file("src/App.tsx", TSX_SAMPLE, min_chars=1)
    assert not parsed.used_fallback and not parsed.has_syntax_errors
    assert symbols(parsed) == {("App", "function", None)}


def test_python_symbols_and_decorators() -> None:
    parsed = chunk_file("pkg/client.py", PY_SAMPLE, min_chars=1)
    assert not parsed.used_fallback and not parsed.has_syntax_errors
    assert symbols(parsed) == {
        ("fetch", "function", None),
        ("Client", "class", None),
        ("__init__", "method", "Client"),
        ("get", "method", "Client"),
    }
    fetch = next(c for c in parsed.chunks if c.symbol == "fetch")
    assert fetch.content.startswith("@decorator")  # the decorator travels with the function
    constant = next(c for c in parsed.chunks if c.symbol == "MAX_RETRIES")
    assert constant.kind == "variable"
    assert_covers(parsed, PY_SAMPLE)


def test_python_large_class_is_split_by_method() -> None:
    parsed = chunk_file("pkg/client.py", PY_SAMPLE, max_chars=80, min_chars=1)
    get = next(c for c in parsed.chunks if c.symbol == "get")
    assert get.kind == "method" and get.parent_symbol == "Client"
    assert_covers(parsed, PY_SAMPLE)
