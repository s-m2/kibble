from importlib.metadata import version

from fastapi.testclient import TestClient

from kibble_engine import __version__
from kibble_engine.api.app import create_app


def test_version_matches_package_metadata() -> None:
    assert version("kibble-engine") == __version__


def test_health_without_auth() -> None:
    client = TestClient(create_app(token=None))
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "version": __version__}


def test_health_requires_token_when_configured() -> None:
    client = TestClient(create_app(token="s3cret"))
    assert client.get("/health").status_code == 401
    assert client.get("/health", headers={"Authorization": "Bearer wrong"}).status_code == 401
    ok = client.get("/health", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200
