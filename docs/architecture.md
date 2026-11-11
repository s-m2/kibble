# Architecture

```
 coding agent (Claude Code, Cursor, ...)
        │  MCP over stdio
        ▼
 @kibble-ai/mcp  (TypeScript)         tool schemas, launcher, generated API client
        │  HTTP, 127.0.0.1 only, per-session bearer token
        ▼
 kibble-engine  (Python, FastAPI)  thin wrapper over core
        │
        ▼
 core: chunking · embeddings · hybrid retrieval · refresh · freshness
        │
        ▼
 Store (SQLite: sqlite-vec + FTS5 + graph tables)   .kibble/index.db
```

## Request flow (search)

1. Stat sweep of the manifest plus a listing of tracked files (new files, deletes, edits).
2. Dirty files are re-chunked, re-embedded (cache keyed by chunk hash) and swapped in
   through `replace_file`. Above a cap, return current results flagged `stale` and
   refresh in the background.
3. Embed the query; run vector KNN and FTS5 BM25; fuse by rank (RRF).
4. (Graph PoC) expand top hits with callers and callees.
5. Respond with results plus freshness metadata.

## Phases

- **PoC 1: vector.** Chunking, embeddings, hybrid retrieval, freshness, eval harness,
  thin end-to-end slice through FastAPI and the MCP shim.
- **PoC 2: graph.** Symbols and edges refreshed inside the same `replace_file`;
  hybrid vector + graph retrieval compared in the eval.
- **Roadmap.** See `roadmap.md`.

Decisions and their reasoning live in `decisions/`.
