# 0012. Hybrid retrieval: weighted RRF with measurable switches

- Status: accepted (design); every default below is unvalidated until the eval runs
- Date: 2026-11-15

## Context
Embeddings find intent but are weak on rare exact identifiers; BM25 is the reverse. Their
scores are incomparable. Retrieval quality is the product, so each choice must be something
the eval harness can switch off and measure.

## Decision
- **Fusion by rank**, weighted reciprocal rank fusion with the standard `k = 60`:
  `score = sum(weight / (k + rank))`. Ties break by best rank, then id (deterministic).
- **Query shape** (identifier / natural / mixed, by simple heuristics) scales the two
  weights: identifier queries favour keywords (x1.4 vs x0.6), natural-language questions
  favour vectors (x1.2 vs x0.8). Switch: `shape_weighting`.
- **Keyword queries are prepared**: stopwords dropped from questions of 3+ words;
  `camelCase` / `snake_case` tokens also searched as their parts. Switches:
  `drop_stopwords`, `expand_identifiers`.
- **Filters apply to each list before fusion**, so ranks reflect what the caller wants;
  candidates are over-fetched 3x when filters are active.
- **File tags** (`test` x0.8, `vendor` x0.5, `generated` x0.5) down-weight noise, except
  that a query mentioning tests/specs removes the test penalty.
- **Diversity**: at most 3 hits per file; chunks of one file overlapping by 50% or more of
  the shorter one collapse. Switches: `max_per_file`, `collapse_overlaps`.
- **Degradation, not failure**: if the embedder is unreachable, keyword results are
  returned with a `degraded` note. An empty query returns an empty, annotated response.
- **Freshness first**: search refreshes the index within a budget and returns the
  freshness report (ADR 0011) with the results. `fresh=False` exists for the eval and
  tooling, not for agents.
- **Explainable results**: every hit carries its rank and score in each list, the response
  carries the weights and keyword terms used, and per-stage timings.
- `--mode vector|keyword` and the config switches exist so the eval can compare
  vector-only, keyword-only and hybrid, and ablate each feature above.

## Open questions (for the eval, not assumptions)
- **No relevance floor.** Vector search always returns its nearest chunks, so a query with
  no good answer still returns `k` results. A minimum similarity, or a rule such as "require
  lexical support for vector-only hits", may be needed; measure how often it matters.
- **All weights, multipliers, `k`, candidate counts and tag penalties are guesses.**
- **BM25 column weights** (symbol terms 5x body) and **path words** (the embedder sees the
  file path, the keyword index doesn't) are untested.
- **Reranking** (cross-encoder or LLM over the top 30-50) is the likely next step if fusion
  plateaus; it is not built.
- Query-shape detection is crude; a misclassified query gets mildly wrong weights, not a
  broken result.

## Consequences
One code path serves the CLI, the future HTTP API and the eval. Results are explainable
enough to debug a bad ranking from the output alone. Because nothing here is validated,
the README must not claim hybrid beats either signal until the eval says so.
