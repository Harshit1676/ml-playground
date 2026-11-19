import pytest
from fastapi.testclient import TestClient

from app.main import create_app



def make_client(monkeypatch: pytest.MonkeyPatch, *, environment: str, origins: str | None = None) -> TestClient:
    monkeypatch.setenv("APP_ENV", environment)
    if origins is None:
        monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
    else:
        monkeypatch.setenv("CORS_ALLOWED_ORIGINS", origins)
    return TestClient(create_app())


def test_health_returns_ok() -> None:
    response = TestClient(create_app()).get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_allows_local_development_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_client(monkeypatch, environment="development")
    response = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "GET" in response.headers["access-control-allow-methods"]


def test_health_allows_configured_production_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_client(
        monkeypatch,
        environment="production",
        origins="https://playground.example.com, https://preview.example.com",
    )
    response = client.get(
        "/api/v1/health",
        headers={"Origin": "https://playground.example.com"},
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://playground.example.com"


def test_health_rejects_unconfigured_production_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_client(
        monkeypatch,
        environment="production",
        origins="https://playground.example.com",
    )
    response = client.options(
        "/api/v1/health",
        headers={
            "Origin": "https://untrusted.example.net",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_production_origin_allows_post_preflight_and_content_type(monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_client(
        monkeypatch,
        environment="production",
        origins="https://playground.example.com",
    )
    response = client.options(
        "/api/v1/experiments/linear-regression",
        headers={
            "Origin": "https://playground.example.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://playground.example.com"
    assert "POST" in response.headers["access-control-allow-methods"]
    assert "content-type" in response.headers["access-control-allow-headers"].lower()


def test_unlisted_production_origin_is_not_granted_post_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    client = make_client(
        monkeypatch,
        environment="production",
        origins="https://playground.example.com",
    )
    response = client.options(
        "/api/v1/experiments/linear-regression",
        headers={
            "Origin": "https://untrusted.example.net",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_production_requires_explicit_origins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)

    with pytest.raises(RuntimeError, match="must be set in production"):
        create_app()


def test_wildcard_origin_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://*.example.com")

    with pytest.raises(RuntimeError, match="Wildcard CORS origins"):
        create_app()
