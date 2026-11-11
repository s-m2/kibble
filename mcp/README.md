# @kibble-ai/mcp

The TypeScript half of [Kibble](https://github.com/s-m2/kibble): an MCP server that
exposes codebase-intelligence tools to coding agents, and launches the Python engine
(`kibble-engine`) as a local sidecar.

> Pre-alpha placeholder release. Nothing useful here yet.

## Development

The launcher normally starts the engine with `uvx`. During development, point it at
your working tree instead:

```bash
export KIBBLE_ENGINE_CMD="uv run --project ../engine kibble-engine serve"
```
