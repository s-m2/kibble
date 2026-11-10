# Related work

Snapshot from public READMEs and comparison pages as of 2026-10-31. Descriptions are
brief and may be out of date; check each project before relying on them.

The space is crowded. A few representative projects:

- **Claude Context (Zilliz):** hybrid BM25 + dense vector code search over MCP.
- **Serena:** language-server and tree-sitter based symbol navigation and edits.
- **cocoindex-code:** tree-sitter AST chunks with local embeddings.
- **codebase-memory-mcp:** knowledge-graph code intelligence in a single binary.
- **code-context (infino):** one local index with keyword, semantic and SQL search.
- **Sourcegraph, Augment Context Engine, Greptile:** commercial/enterprise offerings.

Several projects already address freshness (query-time polling, hooks, stale flags)
and some ship evals, so these are not unique ideas. What Kibble aims to add:
a published, reproducible evaluation of its retrieval and freshness choices
(including distractor volume and "edit then query immediately" cases), atomic
per-file updates across text, vectors and graph, and a pluggable OpenAI-compatible
embedder for hosted, LAN and local setups.
