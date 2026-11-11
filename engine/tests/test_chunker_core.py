"""Chunking logic tested with hand-built syntax trees (no tree-sitter needed).

The fake trees follow the TypeScript grammar's shape, so these tests also pin down
the assumptions the TS adapter makes. The real grammars are checked separately in
test_chunker_grammars.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pytest

from kibble_engine.parsing import chunker
from kibble_engine.parsing.chunker import chunk_file, chunk_tree
from kibble_engine.parsing.languages import LANGUAGES
from kibble_engine.parsing.models import ParsedFile, SymbolSpec

TS = next(spec for spec in LANGUAGES if spec.name == "typescript")


@dataclass
class FakeNode:
    type: str
    start_byte: int
    end_byte: int
    children: list[FakeNode] = field(default_factory=list)
    fields: dict[str, FakeNode] = field(default_factory=dict)
    has_error: bool = False

    def child_by_field_name(self, name: str) -> FakeNode | None:
        return self.fields.get(name)


@dataclass
class T:
    """Tree spec: node type, its exact source text, children, and field->child-index."""

    type: str
    text: str
    kids: tuple[T, ...] = ()
    fields: dict[str, int] = field(default_factory=dict)


def tree(type_: str, text: str, *kids: T, **fields: int) -> T:
    return T(type_, text, kids, dict(fields))


def build(source: bytes, spec: T, lo: int, hi: int) -> FakeNode:
    start = source.index(spec.text.encode(), lo, hi)
    end = start + len(spec.text.encode())
    node = FakeNode(spec.type, start, end)
    pos = start
    for kid in spec.kids:
        child = build(source, kid, pos, end)
        node.children.append(child)
        pos = child.end_byte
    node.fields = {name: node.children[i] for name, i in spec.fields.items()}
    return node


def parse(source: str, *tops: T, path: str = "src/a.ts", **options: Any) -> ParsedFile:
    data = source.encode()
    root = FakeNode("program", 0, len(data))
    pos = 0
    for top in tops:
        node = build(data, top, pos, len(data))
        root.children.append(node)
        pos = node.end_byte
    return chunk_tree(root, data, TS, path, **options)


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def assert_covers(parsed: ParsedFile, source: str) -> None:
    """Every non-whitespace character of the source appears in exactly one chunk."""
    assert squash("".join(c.content for c in parsed.chunks)) == squash(source)


def function_node(name: str, body: str) -> T:
    text = f"function {name}() {body}"
    return tree(
        "function_declaration",
        text,
        tree("function", "function"),
        tree("identifier", name),
        tree("formal_parameters", "()"),
        tree("statement_block", body),
        name=1,
    )


# ---------------------------------------------------------------- definitions


def test_export_wrapper_and_leading_comment_belong_to_the_definition() -> None:
    src = "/** Adds. */\nexport function add(a, b) {\n  return a + b;\n}\n"
    inner = tree(
        "function_declaration",
        "function add(a, b) {\n  return a + b;\n}",
        tree("function", "function"),
        tree("identifier", "add"),
        tree("formal_parameters", "(a, b)"),
        tree("statement_block", "{\n  return a + b;\n}"),
        name=1,
    )
    parsed = parse(
        src,
        tree("comment", "/** Adds. */"),
        tree(
            "export_statement",
            inner.text.join(["export ", ""]),
            tree("export", "export"),
            inner,
            declaration=1,
        ),
    )
    assert len(parsed.chunks) == 1
    chunk = parsed.chunks[0]
    assert (chunk.kind, chunk.symbol, chunk.start_line, chunk.end_line) == ("function", "add", 1, 4)
    assert chunk.content.startswith("/** Adds. */")
    assert "Symbol: add" in chunk.embed_text and "File: src/a.ts" in chunk.embed_text
    assert parsed.symbols == [SymbolSpec("add", "function", 2, 4, 0, None)]
    assert_covers(parsed, src)


def test_non_adjacent_comment_stays_separate() -> None:
    src = "// license\n\n\nfunction f() {}\n"
    parsed = parse(src, tree("comment", "// license"), function_node("f", "{}"), min_chars=1)
    assert [c.kind for c in parsed.chunks] == ["comment", "function"]
    assert_covers(parsed, src)


def test_small_class_is_one_chunk_but_methods_are_symbols() -> None:
    src = "class A {\n  foo() {}\n  bar() {}\n}\n"
    method = lambda n: tree(  # noqa: E731
        "method_definition", f"{n}() {{}}", tree("property_identifier", n), name=0
    )
    cls = tree(
        "class_declaration",
        "class A {\n  foo() {}\n  bar() {}\n}",
        tree("class", "class"),
        tree("type_identifier", "A"),
        tree(
            "class_body",
            "{\n  foo() {}\n  bar() {}\n}",
            tree("{", "{"),
            method("foo"),
            method("bar"),
            tree("}", "}"),
        ),
        name=1,
        body=2,
    )
    parsed = parse(src, cls)
    assert len(parsed.chunks) == 1 and parsed.chunks[0].kind == "class"
    assert [(s.name, s.kind, s.parent, s.chunk_index) for s in parsed.symbols] == [
        ("A", "class", None, 0),
        ("foo", "method", "A", 0),
        ("bar", "method", "A", 0),
    ]
    assert "A foo bar" in parsed.chunks[0].symbol_terms.lower().replace("a foo", "A foo")
    assert_covers(parsed, src)


def test_large_class_splits_into_header_members_and_tail() -> None:
    src = "class A {\n  foo() { return 1; }\n  bar() { return 2; }\n}\n"
    method = lambda n, v: tree(  # noqa: E731
        "method_definition", f"{n}() {{ return {v}; }}", tree("property_identifier", n), name=0
    )
    cls = tree(
        "class_declaration",
        src.strip(),
        tree("class", "class"),
        tree("type_identifier", "A"),
        tree(
            "class_body",
            src[src.index("{") :].strip(),
            tree("{", "{"),
            method("foo", 1),
            method("bar", 2),
            tree("}", "}"),
        ),
        name=1,
        body=2,
    )
    parsed = parse(src, cls, max_chars=40, min_chars=1)
    assert [c.kind for c in parsed.chunks] == ["class", "method", "method", "statements"]
    assert [c.symbol for c in parsed.chunks] == ["A", "foo", "bar", None]
    assert [c.parent_symbol for c in parsed.chunks] == [None, "A", "A", "A"]
    assert parsed.chunks[1].content == "  foo() { return 1; }"  # indentation preserved
    assert [(s.name, s.chunk_index) for s in parsed.symbols] == [("A", 0), ("foo", 1), ("bar", 2)]
    assert_covers(parsed, src)


def test_member_separators_are_not_lost() -> None:
    src = "class A {\n  x = 1;\n  foo() { return 1; }\n}\n"
    field_ = tree("public_field_definition", "x = 1", tree("property_identifier", "x"), name=0)
    method = tree(
        "method_definition", "foo() { return 1; }", tree("property_identifier", "foo"), name=0
    )
    cls = tree(
        "class_declaration",
        src.strip(),
        tree("class", "class"),
        tree("type_identifier", "A"),
        tree(
            "class_body",
            src[src.index("{") :].strip(),
            tree("{", "{"),
            field_,
            tree(";", ";"),
            method,
            tree("}", "}"),
        ),
        name=1,
        body=2,
    )
    parsed = parse(src, cls, max_chars=30, min_chars=1)
    prop = next(c for c in parsed.chunks if c.symbol == "x")
    assert prop.kind == "property" and prop.content.endswith(";")
    assert all(s.name != "x" for s in parsed.symbols)  # properties aren't graph symbols
    assert_covers(parsed, src)


def test_arrow_function_const_is_a_function_but_plain_const_is_a_variable() -> None:
    src = "const f = () => 1;\nconst n = 2;\n"

    def decl(text: str, name: str, value_type: str, value: str) -> T:
        return tree(
            "lexical_declaration",
            text,
            tree("const", "const"),
            tree(
                "variable_declarator",
                f"{name} = {value}",
                tree("identifier", name),
                tree(value_type, value),
                name=0,
                value=1,
            ),
        )

    parsed = parse(
        src,
        decl("const f = () => 1;", "f", "arrow_function", "() => 1"),
        decl("const n = 2;", "n", "number", "2"),
        min_chars=1,
    )
    assert [(c.kind, c.symbol) for c in parsed.chunks] == [("function", "f"), ("variable", "n")]
    assert [s.name for s in parsed.symbols] == ["f"]


# ------------------------------------------------------------ merge and split


def test_small_units_merge_into_one_block() -> None:
    src = "import x from 'x';\nconst a = 1;\nfunction f() {}\n"
    parsed = parse(
        src,
        tree("import_statement", "import x from 'x';"),
        tree(
            "lexical_declaration",
            "const a = 1;",
            tree("const", "const"),
            tree(
                "variable_declarator",
                "a = 1",
                tree("identifier", "a"),
                tree("number", "1"),
                name=0,
                value=1,
            ),
        ),
        function_node("f", "{}"),
    )
    assert len(parsed.chunks) == 1
    chunk = parsed.chunks[0]
    assert chunk.kind == "block" and chunk.symbol is None
    assert "Contains: a, f" in chunk.embed_text
    assert {"a", "f"} <= set(chunk.symbol_terms.split())
    assert [(s.name, s.chunk_index) for s in parsed.symbols] == [("f", 0)]
    assert_covers(parsed, src)


def test_merging_does_not_cross_parents_or_exceed_max() -> None:
    bodies = ["{ return 1; }", "{ return 2; }", "{ return 3; }"]
    src = "\n".join(f"function f{i}() {b}" for i, b in enumerate(bodies)) + "\n"
    parsed = parse(
        src, *[function_node(f"f{i}", b) for i, b in enumerate(bodies)], max_chars=60, min_chars=300
    )
    assert len(parsed.chunks) == 2  # f0+f1 span 57 bytes and merge; adding f2 would exceed 60
    assert all(len(c.content) <= 60 for c in parsed.chunks)
    assert_covers(parsed, src)


def test_merging_does_not_cross_a_parent_boundary() -> None:
    src = "class A {\n  foo() { return 1; }\n  bar() { return 2; }\n}\nfunction f() {}\n"
    method = lambda n, v: tree(  # noqa: E731
        "method_definition", f"{n}() {{ return {v}; }}", tree("property_identifier", n), name=0
    )
    cls_text = src[: src.index("function")].strip()
    cls = tree(
        "class_declaration",
        cls_text,
        tree("class", "class"),
        tree("type_identifier", "A"),
        tree(
            "class_body",
            cls_text[cls_text.index("{") :],
            tree("{", "{"),
            method("foo", 1),
            method("bar", 2),
            tree("}", "}"),
        ),
        name=1,
        body=2,
    )
    parsed = parse(src, cls, function_node("f", "{}"), max_chars=40, min_chars=300)
    last = parsed.chunks[-1]
    assert last.content == "function f() {}" and last.parent_symbol is None
    assert_covers(parsed, src)


def test_oversized_function_is_split_into_parts() -> None:
    stmts = ["a();", "b();", "c();", "d();"]
    body = "{\n" + "".join(f"  {s}\n" for s in stmts) + "}"
    src = f"function big() {body}\n"
    block = tree(
        "statement_block",
        body,
        tree("{", "{"),
        *[tree("expression_statement", s) for s in stmts],
        tree("}", "}"),
    )
    fn = tree(
        "function_declaration",
        f"function big() {body}",
        tree("function", "function"),
        tree("identifier", "big"),
        tree("formal_parameters", "()"),
        block,
        name=1,
    )
    parsed = parse(src, fn, max_chars=30, min_chars=1)

    assert len(parsed.chunks) > 1
    assert all(c.kind == "function" and c.symbol == "big" for c in parsed.chunks)
    n = len(parsed.chunks)
    assert [f"Part: {i + 1}/{n}" in c.embed_text for i, c in enumerate(parsed.chunks)] == [True] * n
    assert all(len(c.content) <= 36 for c in parsed.chunks)
    assert [s.name for s in parsed.symbols] == ["big"]  # one symbol, on the first part
    assert parsed.symbols[0].chunk_index == 0
    assert_covers(parsed, src)


def test_childless_oversized_node_is_split_at_lines_and_utf8_boundaries() -> None:
    text = ("é" * 40) + "\n" + ("é" * 40)
    src = f"const s = `{text}`;\n"
    lit = tree("template_string", f"`{text}`")
    parsed = parse(
        src, tree("expression_statement", f"const s = {lit.text};", lit), max_chars=33, min_chars=1
    )
    assert len(parsed.chunks) > 1
    assert all("\ufffd" not in c.content for c in parsed.chunks)
    assert_covers(parsed, src)


def test_syntax_error_flag_is_reported() -> None:
    data = b"function f() {}\n"
    root = FakeNode("program", 0, len(data), has_error=True)
    parsed = chunk_tree(root, data, TS, "a.ts")
    assert parsed.has_syntax_errors and parsed.chunks == []


def test_chunk_hash_depends_on_path_and_content() -> None:
    src = "function f() {}\n"
    a = parse(src, function_node("f", "{}"), path="src/a.ts")
    b = parse(src, function_node("f", "{}"), path="src/b.ts")
    assert a.chunks[0].content == b.chunks[0].content
    assert a.chunks[0].chunk_hash != b.chunks[0].chunk_hash  # moved file re-embeds


def test_invalid_sizes_are_rejected() -> None:
    with pytest.raises(ValueError, match="max_chars"):
        parse("", max_chars=0)


# ----------------------------------------------------------------- text fallback


def test_unknown_extension_uses_paragraph_fallback() -> None:
    text = "alpha beta\n\nsecond paragraph\n\n" + "line one of a long paragraph\n" * 10
    parsed = chunk_file("notes/readme.xyz", text, max_chars=60, min_chars=20)
    assert parsed.used_fallback and parsed.language == "text"
    assert all(c.kind == "text" for c in parsed.chunks)
    assert all(len(c.content) <= 60 for c in parsed.chunks)
    assert parsed.symbols == []
    assert_covers(parsed, text)


def test_missing_grammar_falls_back_instead_of_failing(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_grammar(*args: object, **kwargs: object) -> None:
        raise ImportError("no tree_sitter_typescript")

    monkeypatch.setattr(chunker, "_parse", no_grammar)
    parsed = chunk_file("a.ts", "function f() {}\n")
    assert parsed.used_fallback and len(parsed.chunks) == 1


def test_empty_file_has_no_chunks() -> None:
    assert chunk_file("empty.txt", "").chunks == []
    assert chunk_file("blank.txt", "\n\n  \n").chunks == []
