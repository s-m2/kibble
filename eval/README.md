# eval

The harness that decides which retrieval choices actually earn their place.
It calls `kibble_engine.core` directly (no server needed).

Planned grid:

- fixed-size vs AST chunks
- raw vs enriched embed text
- vector-only vs BM25-only vs fused (RRF)
- vector-only vs hybrid vs graph-expanded
- **distractor volume**: same questions against a repo plus a pile of unrelated code
- **freshness**: edit then query immediately; create a file then query immediately;
  idle refresh disabled so only the query-time check can save us

`questions/` holds question sets with expected answers. Real repos are fetched by
`scripts/` at pinned SHAs into `eval/.repos/` (gitignored); never vendored.
Results go in `results/` and are summarized in `docs/eval-results.md`.
