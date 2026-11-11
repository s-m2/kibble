# 0009. AST-aware chunking with a small, grammar-independent core

- Status: accepted
- Date: 2026-11-02

## Context
Chunk quality is the biggest lever on retrieval quality. Fixed-size windows cut
functions in half and blur two ideas into one embedding. We also want adding a
language to be a small, low-risk change, and the chunking logic to be testable
without parsing real code.

## Decision
- Walk the tree-sitter tree directly (`type`, `children`, `child_by_field_name`,
  byte offsets) instead of using tree-sitter queries. The query API has changed
  between binding releases; the node surface has not.
- Split the work: `chunker.py` is grammar-independent and talks to a tiny
  `SyntaxNode` protocol; `languages.py` holds per-language adapters (`unwrap`,
  `describe`, `members`). Adding a language is one adapter plus a grammar dependency.
- One unit per definition, with adjacent leading comments and wrappers (`export`,
  decorators) attached. Everything else becomes "statements" units, so every
  non-whitespace byte lands in exactly one chunk (tested as an invariant).
- A class that fits `max_chars` is one chunk (its methods are still recorded as
  symbols); a larger class becomes a header plus one unit per member.
- Oversized units are flattened to their smallest sub-nodes and packed into parts
  (`Part: 2/3`); only the first part carries the symbol. Leaf nodes too big to split
  structurally are split at line boundaries.
- Small adjacent units under the same parent are merged up to `max_chars`.
- Each chunk's embedded text is enriched with path, language, kind and qualified
  symbol; the raw content is stored separately. `symbol_terms` holds identifiers plus
  their split words for BM25. Sizes are UTF-8 bytes, a cheap proxy for tokens.
- Unsupported languages, non-code files, and missing grammars fall back to paragraph
  chunking, so indexing never fails because of a language.

## Consequences
Retrieval units are coherent and carry their context. The defaults (`max_chars=2400`,
`min_chars=300`), merging, and "class as one chunk" are guesses that the eval harness
should settle (see eval/README.md). The adapters encode assumptions about grammar node
names, verified by tests/test_chunker_grammars.py; `scripts/dump_ast.py` shows the real
tree when a grammar upgrade breaks one. Not yet handled: path words in the FTS index
(only the embedder sees the path), call/import edge extraction (Graph PoC), minified or
generated files (the file scanner should exclude them), and TS namespaces/overload
signatures beyond the basics.
