# 0002. SQLite as the single store

- Status: accepted
- Date: 2026-10-31

## Context
Needs: zero install, one transaction per file refresh, small scale (thousands of
files), keyword search for identifiers, shallow graph queries.

## Decision
SQLite with sqlite-vec (vectors), FTS5 (BM25) and plain tables with recursive CTEs
(graph). Retrieval depends on a small `Store` protocol, not on SQLite.

## Consequences
Atomic multi-table refresh and a single-file index. sqlite-vec is brute-force and
pre-1.0; fine to the low hundreds of thousands of chunks. Extension loading can fail
on some Python builds, so detect and report it. Postgres + pgvector, an ANN store or a
graph database can be added behind `Store` if scale or team use demands it.
Alternatives rejected for the default: LanceDB/Chroma (split stores, no shared
transaction), Neo4j (server, adoption friction), PGlite (cannot embed in Python).
