# Plain make on purpose: no extra tool to install, and every target is a
# command you could type yourself. CI calls these same targets.

.PHONY: setup test lint typecheck check openapi check-openapi check-versions dev-engine clean

setup:            ## install both toolchains
	uv sync --all-packages
	cd mcp && npm install

test:             ## run all tests
	uv run pytest
	cd mcp && npm test

lint:             ## lint + format check (Python)
	uv run ruff check .
	uv run ruff format --check .

typecheck:        ## static type checks
	uv run pyright
	cd mcp && npm run typecheck

check-versions:   ## engine and mcp versions must match (lockstep releases)
	uv run python scripts/check_versions.py

openapi:          ## regenerate openapi.json from the FastAPI app
	uv run python scripts/dump_openapi.py > openapi.json

check-openapi:    ## fail if committed openapi.json is stale
	uv run python scripts/dump_openapi.py | diff -u openapi.json - \
	  || (echo "openapi.json is stale: run 'make openapi' and commit" && exit 1)

check: lint typecheck check-versions check-openapi test   ## everything CI runs

dev-engine:       ## run the engine against the working tree, no auth, fixed port
	uv run kibble-engine serve --no-auth --port 8765

clean:
	rm -rf .venv .pytest_cache .ruff_cache mcp/dist mcp/node_modules
