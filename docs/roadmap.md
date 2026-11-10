# Roadmap

## PoC 1: vector
- [ ] Schema, `SqliteStore`, `replace_file`, crash-atomicity test
- [ ] OpenAI-compatible embedder, `embedder_id`, `doctor` command, fake test server
- [ ] Tree-sitter chunker (TS/JS first), enriched embed text, fixtures
- [ ] Manifest, stat sweep, `refresh_files`, freshness metadata
- [ ] Hybrid search (sqlite-vec + FTS5 + RRF)
- [ ] Eval harness: chunking, fusion, distractor volume, freshness cases
- [ ] FastAPI endpoints + job model for indexing
- [ ] MCP shim: generated client, launcher lifecycle, tools

## PoC 2: graph
- [ ] Symbols and edges per file inside `replace_file`
- [ ] Resolution pass; traversal limits and truncation flags
- [ ] `find_callers`, `get_dependencies`, `impact_of_change`
- [ ] Hybrid vector + graph retrieval in the eval
- [ ] Optional `refresh_index` tool and agent skill (a nudge, not a guarantee)

## Beyond the PoC (product work)
- Claude Code hooks (deterministic turn-end / post-edit refresh)
- Filesystem watcher (watchdog), git hooks, HEAD-change refresh
- Checkpoint integrity checks
- TypeScript compiler-API resolver for precise call and import edges
- Reranking, result diversity (MMR)
- Offline bundle and local embedder distribution (internal mirrors, vendored weights)
- Optional LLM features (`explain_module`, reranker) behind `[llm]` config
- Shared daemon + Postgres/pgvector backend for team use
- Published packages with provenance; smoke tests under npx, bunx, pnpm dlx
- Scaling tiers: lower dimensions, quantization, candidate narrowing, ANN backend
