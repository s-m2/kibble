# Contributing

Kibble is pre-alpha. Issues and discussion are welcome; check the roadmap
(`docs/roadmap.md`) before proposing large changes.

## Setup

Requirements: [uv](https://docs.astral.sh/uv/), Node 22+ (see `.nvmrc`), `make`.

```bash
make setup     # installs both toolchains
make check     # lint, typecheck, version + OpenAPI drift checks, tests
```

## Ground rules

- Read `AGENTS.md`; it is the short list of architecture rules, and it applies to
  humans too.
- Architecture decisions go in `docs/decisions/` as a short ADR.
- Changing the HTTP API? Run `make openapi` and commit `openapi.json`; CI fails on drift.
- Tests must not touch the network. Embedding calls go to the fake server in the test suite.
- One change, one PR; keep Python and TypeScript changes for a feature together.
