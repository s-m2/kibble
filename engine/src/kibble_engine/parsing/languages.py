"""Language adapters: the only place that knows tree-sitter node type names.

Each adapter answers three questions about a syntax node, using plain tree walking
(no tree-sitter queries, whose binding API has changed between releases):

- `unwrap`:   if this node is a wrapper (`export ...`, a Python decorator), which
              child is the real definition?
- `describe`: what is it (function, class, ...) and what is it called?
- `members`:  for containers (classes), which children are the members?

Grammar assumptions live here and are checked against real grammars by
tests/test_chunker_grammars.py. If a grammar release renames a node type,
`scripts/dump_ast.py <file>` shows the tree so the fix is a one-line change.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from kibble_engine.parsing.nodes import SyntaxNode


@dataclass(frozen=True, slots=True)
class Described:
    kind: str  # function | class | interface | type | enum | method | property | variable
    name: str | None
    container: bool = False


@dataclass(frozen=True)
class LanguageSpec:
    name: str
    extensions: tuple[str, ...]
    load: Callable[[], Any]  # returns a tree_sitter.Language
    comment_types: frozenset[str]
    unwrap: Callable[[SyntaxNode], SyntaxNode]
    describe: Callable[[SyntaxNode, bytes], Described | None]
    members: Callable[[SyntaxNode], Sequence[SyntaxNode]]


def node_text(node: SyntaxNode, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _name_of(node: SyntaxNode, source: bytes, field: str = "name") -> str | None:
    child = node.child_by_field_name(field)
    return node_text(child, source) if child is not None else None


# --------------------------------------------------------- JavaScript / TypeScript

_JS_FUNCTION_VALUES = frozenset(
    {"arrow_function", "function_expression", "function", "generator_function"}
)
_JS_PUNCTUATION = frozenset({"{", "}", ";", ","})


def _js_unwrap(node: SyntaxNode) -> SyntaxNode:
    if node.type == "export_statement":
        return node.child_by_field_name("declaration") or node.child_by_field_name("value") or node
    return node


def _js_describe(node: SyntaxNode, source: bytes) -> Described | None:
    t = node.type
    if t in ("function_declaration", "generator_function_declaration", "function_signature"):
        return Described("function", _name_of(node, source))
    if t in ("function_expression", "arrow_function", "generator_function", "function"):
        # Reached via `export default function () {}` and similar.
        return Described("function", _name_of(node, source) or "default")
    if t in ("class_declaration", "abstract_class_declaration", "class"):
        return Described("class", _name_of(node, source) or "default", container=True)
    if t == "interface_declaration":
        return Described("interface", _name_of(node, source))
    if t == "type_alias_declaration":
        return Described("type", _name_of(node, source))
    if t == "enum_declaration":
        return Described("enum", _name_of(node, source))
    if t in ("lexical_declaration", "variable_declaration"):
        declarator = next((c for c in node.children if c.type == "variable_declarator"), None)
        if declarator is None:
            return None
        name_node = declarator.child_by_field_name("name")
        if name_node is None or name_node.type != "identifier":
            return None  # destructuring etc.
        value = declarator.child_by_field_name("value")
        kind = "function" if value is not None and value.type in _JS_FUNCTION_VALUES else "variable"
        return Described(kind, node_text(name_node, source))
    if t in ("method_definition", "abstract_method_signature", "method_signature"):
        return Described("method", _name_of(node, source))
    if t in ("public_field_definition", "field_definition"):
        name = _name_of(node, source) or _name_of(node, source, "property")
        value = node.child_by_field_name("value")
        kind = "method" if value is not None and value.type in _JS_FUNCTION_VALUES else "property"
        return Described(kind, name)
    return None


def _js_members(node: SyntaxNode) -> Sequence[SyntaxNode]:
    body = node.child_by_field_name("body")
    if body is None:
        return []
    return [c for c in body.children if c.type not in _JS_PUNCTUATION]


# ------------------------------------------------------------------------- Python


def _py_unwrap(node: SyntaxNode) -> SyntaxNode:
    if node.type == "decorated_definition":
        return node.child_by_field_name("definition") or node
    return node


def _py_describe(node: SyntaxNode, source: bytes) -> Described | None:
    t = node.type
    if t == "function_definition":  # async defs share this node type
        return Described("function", _name_of(node, source))
    if t == "class_definition":
        return Described("class", _name_of(node, source), container=True)
    if t == "expression_statement":
        first = node.children[0] if node.children else None
        if first is not None and first.type == "assignment":
            left = first.child_by_field_name("left")
            if left is not None and left.type == "identifier":
                return Described("variable", node_text(left, source))
    return None


def _py_members(node: SyntaxNode) -> Sequence[SyntaxNode]:
    body = node.child_by_field_name("body")
    return list(body.children) if body is not None else []


# ----------------------------------------------------------------------- loaders
# Imports are lazy so a missing grammar package degrades to the text fallback
# instead of breaking `import kibble_engine`.


def _load_python() -> Any:
    import tree_sitter_python
    from tree_sitter import Language

    return Language(tree_sitter_python.language())


def _load_javascript() -> Any:
    import tree_sitter_javascript
    from tree_sitter import Language

    return Language(tree_sitter_javascript.language())


def _load_typescript() -> Any:
    import tree_sitter_typescript
    from tree_sitter import Language

    return Language(tree_sitter_typescript.language_typescript())


def _load_tsx() -> Any:
    import tree_sitter_typescript
    from tree_sitter import Language

    return Language(tree_sitter_typescript.language_tsx())


_COMMENTS = frozenset({"comment"})

LANGUAGES: tuple[LanguageSpec, ...] = (
    LanguageSpec(
        "typescript",
        (".ts", ".mts", ".cts"),
        _load_typescript,
        _COMMENTS,
        _js_unwrap,
        _js_describe,
        _js_members,
    ),
    LanguageSpec("tsx", (".tsx",), _load_tsx, _COMMENTS, _js_unwrap, _js_describe, _js_members),
    LanguageSpec(
        "javascript",
        (".js", ".jsx", ".mjs", ".cjs"),
        _load_javascript,
        _COMMENTS,
        _js_unwrap,
        _js_describe,
        _js_members,
    ),
    LanguageSpec(
        "python", (".py", ".pyi"), _load_python, _COMMENTS, _py_unwrap, _py_describe, _py_members
    ),
)

_BY_EXTENSION = {ext: spec for spec in LANGUAGES for ext in spec.extensions}


def language_for_path(path: str) -> LanguageSpec | None:
    return _BY_EXTENSION.get(PurePosixPath(path).suffix.lower())
