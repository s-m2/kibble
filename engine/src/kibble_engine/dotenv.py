"""A minimal .env reader (no dependency, no expansion, no os.environ mutation).

Rules: `KEY=VALUE` per line, optional leading `export `, `#` comments on their own
line or after whitespace following an unquoted value, matching single or double
quotes around a value are removed. Anything else is ignored. Callers merge the
result *under* the real environment, so exported variables always win.
"""

from __future__ import annotations

import re
from pathlib import Path

_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_dotenv(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not _KEY.match(key):
            continue
        values[key] = _clean_value(value.strip())
    return values


def _clean_value(value: str) -> str:
    if value[:1] in ("'", '"'):
        quote = value[0]
        end = value.find(quote, 1)
        if end != -1:
            return value[1:end]
        return value[1:]
    return re.split(r"\s+#", value, maxsplit=1)[0].rstrip()


def load_dotenv_file(path: Path) -> dict[str, str]:
    try:
        return parse_dotenv(path.read_text(encoding="utf-8"))
    except OSError:
        return {}
