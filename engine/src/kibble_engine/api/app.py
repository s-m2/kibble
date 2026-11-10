"""HTTP app factory.

Security model: the server binds to 127.0.0.1 only (enforced in the CLI) and
requires a per-session bearer token supplied by the launcher via KIBBLE_TOKEN.
Localhost is not a trust boundary on its own.
"""

from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from kibble_engine import __version__

_bearer = HTTPBearer(auto_error=False)


class Health(BaseModel):
    status: str
    version: str


def create_app(token: str | None = None) -> FastAPI:
    """Build the app. `token=None` disables auth (tests and `--no-auth` dev only)."""
    app = FastAPI(title="Kibble engine", version=__version__)

    def require_token(
        creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    ) -> None:
        if token is None:
            return
        supplied = creds.credentials if creds else ""
        if not hmac.compare_digest(supplied.encode(), token.encode()):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="invalid or missing token")

    @app.get("/health", response_model=Health, dependencies=[Depends(require_token)])
    def health() -> Health:
        return Health(status="ok", version=__version__)

    return app
