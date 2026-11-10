# 0001. One monorepo, three packages

- Status: accepted
- Date: 2026-10-31

## Context
Kibble has a Python engine, a TypeScript MCP shim, and an eval harness. The shim
launches an exact matching engine version, and the HTTP contract between them must
not drift.

## Decision
One repository: `engine/` (PyPI), `mcp/` (npm), `eval/` (unpublished). A uv workspace
covers the Python packages; `mcp/` is a plain npm package. One CI pipeline; releases
are lockstep.

## Consequences
A change to the API touches the engine, `openapi.json`, the generated client and the
MCP tool in one PR. We carry two toolchains in one repo, which is inherent to the
architecture. If packages later need independent cadences, prefixed tags can split
releases, and directories can be extracted with history.
