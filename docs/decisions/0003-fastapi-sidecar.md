# 0003. FastAPI sidecar over stdio

- Status: accepted
- Date: 2026-10-31

## Context
The TS MCP server needs the Python engine at query time (query embedding, retrieval).
Options: a stdio child with a custom protocol, or a local HTTP service.

## Decision
The engine is a FastAPI app on 127.0.0.1 with a per-session bearer token. The TS
client is generated from the committed `openapi.json`. Long work (indexing) is a job
with polling.

## Consequences
Typed contract, easy debugging (curl, `/docs`), standard error and timeout semantics.
We take on lifecycle management (free port, health polling, orphan cleanup, Windows)
and localhost security (bind loopback only, token). One engine per index, enforced
with a lockfile.
