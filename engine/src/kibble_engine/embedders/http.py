"""The one place that chooses the HTTP client library.

httpx2 is the successor to httpx (Starlette's TestClient now prefers it). If you
are still on plain httpx, change this import to `import httpx`; the rest of the
code only uses the small API shared by both.
"""

import httpx2 as httpx

__all__ = ["httpx"]
