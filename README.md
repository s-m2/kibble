# Kibble

**Bite-sized codebase intelligence for coding agents.**

Kibble indexes a repository and gives coding agents (Claude Code, Cursor, and any
MCP client) fast, *fresh* answers about it: where something lives, what calls it,
and what a change might affect. It chunks code along syntax boundaries, combines
vector and keyword retrieval, and (soon) a call/import graph, all in a single local
SQLite file.

> **Status: pre-alpha scaffold.** Nothing works yet; this repo is being built in the
> open. See [`docs/roadmap.md`](docs/roadmap.md).

## Design in one minute

- **Chunk by syntax, not by size.** tree-sitter boundaries, with path and parent
  context embedded alongside the code.
- **Hybrid retrieval.** Vector similarity for intent ("where is auth handled?"), BM25
  for exact identifiers, fused by rank.
- **Never silently stale.** Every query checks the files it depends on, refreshes what
  changed, and reports how fresh the answer is. Watchers and hooks are optimizations,
  not the guarantee.
- **One file, one transaction.** Text, vectors and graph for a file update atomically.
- **Bring your own embeddings.** Any OpenAI-compatible endpoint: hosted, LAN or local.
- **Measured, not claimed.** An eval harness compares chunking, fusion and freshness
  strategies; results are published in [`docs/eval-results.md`](docs/eval-results.md).

```
agent ──MCP──▶ @kibble-ai/mcp (TS) ──HTTP, loopback + token──▶ kibble-engine (Python) ──▶ SQLite
```

Details: [`docs/architecture.md`](docs/architecture.md), decisions in
[`docs/decisions/`](docs/decisions/), prior art in [`docs/related-work.md`](docs/related-work.md).

## Layout

| Path      | What                                   | Published as    |
|-----------|----------------------------------------|-----------------|
| `engine/` | Python engine (core, store, API, CLI)  | `kibble-engine` (PyPI) |
| `mcp/`    | TypeScript MCP server + launcher       | `@kibble-ai/mcp` (npm)    |
| `eval/`   | Retrieval and freshness evaluation     | not published   |

## Development

Requires [uv](https://docs.astral.sh/uv/), Node 22+, and `make`.

```bash
make setup    # install both toolchains
make check    # lint, typecheck, drift checks, tests (what CI runs)
```

First-time lockfile and API generation steps are listed in
[`CONTRIBUTING.md`](CONTRIBUTING.md). Agents working here should read
[`AGENTS.md`](AGENTS.md).

## Privacy

With a hosted embedding endpoint, chunk text is sent to that provider. Use a local or
LAN endpoint to keep code on your machines. Kibble itself sends nothing anywhere else
and has no telemetry. API keys are read from environment variables only.

## Limitations (so far)

Everything. Also by design: single language first (TS/JS), repositories up to a few
thousand files, brute-force vector search, name-based (not type-resolved) call edges
until the resolver lands.

## License

Apache-2.0
