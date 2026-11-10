# AGENTS.md

Instructions for coding agents (and humans) working in this repo. Keep this short;
details live in `docs/` and `docs/decisions/`.

## What this is

Kibble: codebase intelligence for coding agents. A TypeScript MCP server
(`mcp/`, npm `@kibble/mcp`) launches a Python engine (`engine/`, PyPI
`kibble-engine`) that talks HTTP on localhost. `eval/` measures retrieval quality.

## Commands

```bash
make setup      # uv sync + npm install
make check      # lint, typecheck, version + OpenAPI drift, all tests (what CI runs)
make test       # tests only
make openapi    # regenerate openapi.json after changing the HTTP API
```

## Architecture rules (do not break without an ADR)

1. `engine/src/kibble_engine/core/` never imports FastAPI or web code. The API
   layer is a thin wrapper; the CLI and eval call `core` directly.
2. All writes for a file go through one transaction (`Store.replace_file`):
   manifest, chunks, FTS rows, vectors, symbols, edges. No partial updates.
3. Freshness is guaranteed at query time (stat sweep of the manifest plus new-file
   and delete detection). Watchers, hooks and idle refresh are optimizations only.
4. Tool responses carry freshness metadata; never silently serve stale data.
5. The engine binds to 127.0.0.1 and requires the per-session bearer token.
6. Secrets come from environment variables only. Never log request bodies or headers.
7. stdout of the engine carries one handshake line at most; logs go to stderr.
8. Tests never touch the network. Embedding tests use the fake OpenAI-compatible server.
9. Retrieval depends on `store.base.Store`, not on SQLite directly.
10. `openapi.json` is committed; the TS client is generated from it.

## Conventions

- Python 3.11+, ruff, pyright. TypeScript strict, ESM (`.js` extensions in imports).
- Versions of `kibble-engine` and `@kibble/mcp` move in lockstep (`make check-versions`).
- New dependency? Say why in the PR; prefer the standard library when it is close.
- Architecture decision? Add a half-page ADR in `docs/decisions/`.

## Definition of done

`make check` passes, behavior is covered by a test, and any user-visible change is in
`CHANGELOG.md`.
