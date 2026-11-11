"""AST-aware chunking.

Pipeline for one file:

1. Parse (tree-sitter) and walk the top-level nodes into *units*: one per
   definition, with its adjacent leading comments and any wrapper (`export`,
   decorators) attached. Everything else (imports, statements) becomes
   "statements" units, so every non-whitespace byte lands in exactly one unit.
2. Classes: if a class fits in `max_chars` it stays one chunk (its methods are still
   recorded as symbols); otherwise it is split into a header plus one unit per member.
3. Oversized units are flattened into their smallest sub-nodes and packed greedily
   into parts (`Part: 2/3`); a node with no children is split at line boundaries.
4. Small adjacent units under the same parent are merged up to `max_chars`, so
   three-line helpers don't become three near-empty chunks.
5. Each chunk gets enriched `embed_text` (path, language, kind, qualified symbol)
   and `symbol_terms` (identifiers plus their split words) for BM25.

Sizes are measured in bytes of UTF-8 source, a cheap, conservative proxy for tokens.
Files in unsupported languages go through the same steps 3-5 using paragraphs as
units ("text" chunks).
"""

from __future__ import annotations

import bisect
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from kibble_engine.hashing import sha256_hex
from kibble_engine.parsing.identifiers import symbol_terms
from kibble_engine.parsing.languages import LanguageSpec, language_for_path
from kibble_engine.parsing.models import ChunkSpec, ParsedFile, SymbolSpec
from kibble_engine.parsing.nodes import SyntaxNode

log = logging.getLogger(__name__)

DEFAULT_MAX_CHARS = 2400
DEFAULT_MIN_CHARS = 300

# Kinds that become graph symbols (variables and properties are searchable but not nodes).
SYMBOL_KINDS = frozenset({"function", "method", "class", "interface", "type", "enum"})

_LANGUAGE_CACHE: dict[str, Any] = {}


@dataclass
class _Sym:
    name: str
    kind: str
    start: int
    end: int
    parent: str | None


@dataclass
class _Unit:
    start: int
    end: int
    kind: str
    name: str | None
    parent: str | None
    syms: list[_Sym] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    part: tuple[int, int] | None = None

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass
class _Ctx:
    source: bytes
    line_starts: list[int]
    spec: LanguageSpec | None
    max_chars: int
    min_chars: int

    def line_of(self, offset: int) -> int:
        return bisect.bisect_right(self.line_starts, offset)


# ----------------------------------------------------------------------- public


def chunk_file(
    path: str,
    source: bytes | str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> ParsedFile:
    """Chunk one file. Unsupported languages, or a missing grammar, use the text fallback."""
    data = source.encode("utf-8") if isinstance(source, str) else source
    spec = language_for_path(path)
    if spec is not None:
        try:
            root = _parse(spec, data)
        except ImportError as exc:
            log.warning("grammar for %s unavailable (%s); using text fallback", spec.name, exc)
        else:
            return chunk_tree(root, data, spec, path, max_chars=max_chars, min_chars=min_chars)
    return _chunk_text(path, data, max_chars=max_chars, min_chars=min_chars)


def chunk_tree(
    root: SyntaxNode,
    source: bytes,
    spec: LanguageSpec,
    path: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> ParsedFile:
    """Chunk an already-parsed tree. Separate from `chunk_file` so tests can pass fake trees."""
    ctx = _make_ctx(source, spec, max_chars, min_chars)
    units = _build_units(ctx, root.children, None)
    return _emit(ctx, path, spec.name, _merge(ctx, units), root.has_error, False)


# --------------------------------------------------------------------- parsing


def _parse(spec: LanguageSpec, data: bytes) -> SyntaxNode:
    from tree_sitter import Parser  # lazy: keeps the chunking logic importable without it

    language = _LANGUAGE_CACHE.get(spec.name)
    if language is None:
        language = spec.load()
        _LANGUAGE_CACHE[spec.name] = language
    return Parser(language).parse(data).root_node


def _make_ctx(source: bytes, spec: LanguageSpec | None, max_chars: int, min_chars: int) -> _Ctx:
    if max_chars < 1 or min_chars < 0:
        raise ValueError("max_chars must be >= 1 and min_chars >= 0")
    starts = [0] + [m.end() for m in re.finditer(rb"\n", source)]
    return _Ctx(source, starts, spec, max_chars, min_chars)


# ----------------------------------------------------------------------- units


def _build_units(ctx: _Ctx, nodes: Sequence[SyntaxNode], parent: str | None) -> list[_Unit]:
    assert ctx.spec is not None
    units: list[_Unit] = []
    pending: list[SyntaxNode] = []

    def standalone(comment: SyntaxNode) -> _Unit:
        return _Unit(comment.start_byte, comment.end_byte, "comment", None, parent)

    for node in nodes:
        if node.type in ctx.spec.comment_types:
            pending.append(node)
            continue
        attached = _adjacent_tail(ctx, pending, node)
        for comment in pending[: len(pending) - len(attached)]:
            units.append(standalone(comment))
        pending.clear()
        start = attached[0].start_byte if attached else node.start_byte
        units.extend(_units_for_node(ctx, node, start, parent))
    units.extend(standalone(c) for c in pending)
    return units


def _adjacent_tail(ctx: _Ctx, comments: Sequence[SyntaxNode], node: SyntaxNode) -> list[SyntaxNode]:
    """Trailing comments that touch `node` (at most one line break between each)."""
    tail: list[SyntaxNode] = []
    next_start = node.start_byte
    for comment in reversed(comments):
        between = ctx.source[comment.end_byte : next_start]
        if between.strip() or between.count(b"\n") > 1:
            break
        tail.insert(0, comment)
        next_start = comment.start_byte
    return tail


def _units_for_node(ctx: _Ctx, node: SyntaxNode, start: int, parent: str | None) -> list[_Unit]:
    assert ctx.spec is not None
    inner = ctx.spec.unwrap(node)
    described = ctx.spec.describe(inner, ctx.source)
    end = node.end_byte

    if described is None:
        return _fit(ctx, _Unit(start, end, "statements", None, parent), node)

    kind = "method" if parent is not None and described.kind == "function" else described.kind
    name = described.name
    syms = [_Sym(name, kind, node.start_byte, end, parent)] if name and kind in SYMBOL_KINDS else []

    if described.container:
        return _class_units(ctx, node, inner, start, parent, name, syms)

    unit = _Unit(start, end, kind, name, parent, syms, [name] if name else [])
    return _fit(ctx, unit, node)


def _class_units(
    ctx: _Ctx,
    node: SyntaxNode,
    inner: SyntaxNode,
    start: int,
    parent: str | None,
    name: str | None,
    syms: list[_Sym],
) -> list[_Unit]:
    assert ctx.spec is not None
    end = node.end_byte
    qualified = f"{parent}.{name}" if parent and name else (name or parent)
    members = ctx.spec.members(inner)
    names = [name] if name else []

    if end - start <= ctx.max_chars:
        member_syms = _member_symbols(ctx, members, qualified)
        return [_Unit(start, end, "class", name, parent, syms + member_syms, names)]

    member_units = _build_units(ctx, members, qualified)
    _absorb_separators(ctx, member_units, end)
    header_end = member_units[0].start if member_units else end
    header = _Unit(start, header_end, "class", name, parent, syms, names)
    result = _fit(ctx, header, None)
    result.extend(member_units)
    if member_units and member_units[-1].end < end:
        tail = _Unit(member_units[-1].end, end, "statements", None, qualified)
        if ctx.source[tail.start : tail.end].strip():
            result.append(tail)
    return result


def _absorb_separators(ctx: _Ctx, units: Sequence[_Unit], boundary: int) -> None:
    """Attach `;` / `,` that sit between members (outside any member node) to the member
    before them, so no source text is dropped from the chunks."""
    for i, unit in enumerate(units):
        next_start = units[i + 1].start if i + 1 < len(units) else boundary
        match = re.match(rb"[ \t\r\n]*[;,]+", ctx.source[unit.end : next_start])
        if match:
            unit.end += match.end()


def _member_symbols(ctx: _Ctx, members: Sequence[SyntaxNode], qualified: str | None) -> list[_Sym]:
    """Symbols for members of a class that is kept as a single chunk."""
    assert ctx.spec is not None
    out: list[_Sym] = []
    for member in members:
        if member.type in ctx.spec.comment_types:
            continue
        inner = ctx.spec.unwrap(member)
        described = ctx.spec.describe(inner, ctx.source)
        if described is None or not described.name:
            continue
        kind = "method" if described.kind == "function" else described.kind
        if kind in SYMBOL_KINDS:
            out.append(_Sym(described.name, kind, member.start_byte, member.end_byte, qualified))
        if described.container:
            nested = f"{qualified}.{described.name}" if qualified else described.name
            out.extend(_member_symbols(ctx, ctx.spec.members(inner), nested))
    return out


# --------------------------------------------------------------- splitting


def _fit(ctx: _Ctx, unit: _Unit, node: SyntaxNode | None) -> list[_Unit]:
    """Return `unit` if it fits, else split it into parts of at most max_chars."""
    if unit.size <= ctx.max_chars:
        return [unit]
    pieces = _flatten(ctx, node) if node is not None else [(unit.start, unit.end)]
    if not pieces:
        pieces = [(unit.start, unit.end)]
    pieces[0] = (unit.start, pieces[0][1])  # first part keeps leading comments
    pieces[-1] = (pieces[-1][0], unit.end)  # last part keeps the wrapper's tail
    pieces = [r for p in pieces for r in _hard_split(ctx.source, p[0], p[1], ctx.max_chars)]
    ranges = _pack(pieces, ctx.max_chars)
    parts: list[_Unit] = []
    for i, (s, e) in enumerate(ranges):
        first = i == 0
        parts.append(
            _Unit(
                s,
                e,
                unit.kind,
                unit.name,
                unit.parent,
                unit.syms if first else [],
                unit.names if first else [],
                (i + 1, len(ranges)) if len(ranges) > 1 else None,
            )
        )
    return parts


def _flatten(ctx: _Ctx, node: SyntaxNode) -> list[tuple[int, int]]:
    """Smallest-sufficient sub-ranges of `node`, in order, each <= max_chars if possible."""
    if not node.children:
        return _hard_split(ctx.source, node.start_byte, node.end_byte, ctx.max_chars)
    out: list[tuple[int, int]] = []
    for child in node.children:
        if child.end_byte - child.start_byte <= ctx.max_chars:
            out.append((child.start_byte, child.end_byte))
        else:
            out.extend(_flatten(ctx, child))
    return out


def _pack(pieces: Sequence[tuple[int, int]], max_chars: int) -> list[tuple[int, int]]:
    """Greedily join consecutive pieces into contiguous ranges of at most max_chars."""
    ranges: list[tuple[int, int]] = []
    current: tuple[int, int] | None = None
    for s, e in pieces:
        if current is None:
            current = (s, e)
        elif e - current[0] <= max_chars:
            current = (current[0], e)
        else:
            ranges.append(current)
            current = (s, e)
    if current is not None:
        ranges.append(current)
    return ranges


def _hard_split(source: bytes, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    """Split [start, end) at line boundaries where possible, else at UTF-8 boundaries."""
    if end - start <= max_chars:
        return [(start, end)]
    out: list[tuple[int, int]] = []
    pos = start
    while pos < end:
        limit = min(pos + max_chars, end)
        if limit < end:
            newline = source.rfind(b"\n", pos, limit)
            if newline > pos:
                limit = newline + 1
            else:
                while limit > pos + 1 and (source[limit] & 0xC0) == 0x80:
                    limit -= 1
        out.append((pos, limit))
        pos = limit
    return out


# ------------------------------------------------------------------- merging


def _merge(ctx: _Ctx, units: Sequence[_Unit]) -> list[_Unit]:
    """Merge runs of small units (same parent) so tiny definitions don't become tiny chunks."""
    out: list[_Unit] = []
    buffer: list[_Unit] = []

    def flush() -> None:
        if len(buffer) == 1:
            out.append(buffer[0])
        elif buffer:
            out.append(_combine(buffer))
        buffer.clear()

    for unit in units:
        if unit.size >= ctx.min_chars or unit.part is not None:
            flush()
            out.append(unit)
            continue
        if buffer and (
            buffer[0].parent != unit.parent or unit.end - buffer[0].start > ctx.max_chars
        ):
            flush()
        buffer.append(unit)
    flush()
    return out


def _combine(units: Sequence[_Unit]) -> _Unit:
    kind = "text" if all(u.kind == "text" for u in units) else "block"
    names: list[str] = []
    syms: list[_Sym] = []
    for u in units:
        names.extend(u.names)
        syms.extend(u.syms)
    return _Unit(units[0].start, units[-1].end, kind, None, units[0].parent, syms, names)


# -------------------------------------------------------------------- emitting


def _emit(
    ctx: _Ctx,
    path: str,
    language: str,
    units: Sequence[_Unit],
    has_errors: bool,
    used_fallback: bool,
) -> ParsedFile:
    chunks: list[ChunkSpec] = []
    symbols: list[SymbolSpec] = []
    for unit in units:
        start = unit.start
        while start < unit.end and ctx.source[start] in b" \t\r\n":
            start += 1  # drop leading blank lines
        if start >= unit.end:
            continue
        while start > 0 and ctx.source[start - 1] in (0x20, 0x09):
            start -= 1  # keep the first line's indentation
        content = ctx.source[start : unit.end].decode("utf-8", errors="replace").rstrip()
        if not content.strip():
            continue
        start_line = ctx.line_of(start)
        end_line = max(start_line, ctx.line_of(max(unit.end - 1, start)))
        embed_text = _embed_text(path, language, unit, content)
        names = [*unit.names, *(s.name for s in unit.syms)]
        chunk_index = len(chunks)
        chunks.append(
            ChunkSpec(
                chunk_hash=sha256_hex(embed_text),
                kind=unit.kind,
                start_line=start_line,
                end_line=end_line,
                content=content,
                embed_text=embed_text,
                symbol=unit.name if unit.kind in _NAMED_KINDS else None,
                parent_symbol=unit.parent,
                symbol_terms=symbol_terms(*names, unit.parent),
            )
        )
        for sym in unit.syms:
            symbols.append(
                SymbolSpec(
                    sym.name,
                    sym.kind,
                    ctx.line_of(sym.start),
                    max(ctx.line_of(sym.start), ctx.line_of(max(sym.end - 1, sym.start))),
                    chunk_index,
                    sym.parent,
                )
            )
    return ParsedFile(language, chunks, symbols, has_errors, used_fallback)


_NAMED_KINDS = frozenset(
    {"function", "method", "class", "interface", "type", "enum", "variable", "property"}
)


def _embed_text(path: str, language: str, unit: _Unit, content: str) -> str:
    """Context-enriched text for the embedder. The raw content is stored separately."""
    lines = [f"File: {path}", f"Language: {language}", f"Kind: {unit.kind}"]
    if unit.name:
        qualified = f"{unit.parent}.{unit.name}" if unit.parent else unit.name
        lines.append(f"Symbol: {qualified}")
    else:
        if unit.parent:
            lines.append(f"Within: {unit.parent}")
        contained = list(dict.fromkeys([*unit.names, *(s.name for s in unit.syms)]))
        if contained:
            lines.append("Contains: " + ", ".join(contained[:12]))
    if unit.part is not None:
        lines.append(f"Part: {unit.part[0]}/{unit.part[1]}")
    return "\n".join(lines) + "\n\n" + content


# ---------------------------------------------------------------- text fallback


def _chunk_text(path: str, data: bytes, *, max_chars: int, min_chars: int) -> ParsedFile:
    """Paragraph-based chunking for languages without an adapter (and non-code files)."""
    ctx = _make_ctx(data, None, max_chars, min_chars)
    units: list[_Unit] = []
    for match in re.finditer(rb"(?:[^\n]*\S[^\n]*(?:\n|$))+", data):
        unit = _Unit(match.start(), match.end(), "text", None, None)
        units.extend(_fit(ctx, unit, None))
    return _emit(ctx, path, "text", _merge(ctx, units), False, True)
