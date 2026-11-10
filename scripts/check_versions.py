"""Fail unless the engine, MCP package.json and MCP version.ts versions all match."""

import json
import re
import sys
from pathlib import Path

from kibble_engine import __version__

root = Path(__file__).resolve().parent.parent
pkg_version = json.loads((root / "mcp" / "package.json").read_text())["version"]
match = re.search(r'VERSION = "([^"]+)"', (root / "mcp" / "src" / "version.ts").read_text())
ts_version = match.group(1) if match else "<not found>"

versions = {
    "kibble-engine": __version__,
    "mcp/package.json": pkg_version,
    "mcp/src/version.ts": ts_version,
}
if len(set(versions.values())) != 1:
    sys.exit(f"version mismatch: {versions}")
print(f"versions match: {__version__}")
