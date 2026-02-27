"""Tests for the FastAPI API endpoints (without starting the engine)."""

import pytest
from fastapi.testclient import TestClient

from qe.api.app import app


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


def test_health_endpoint(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "timestamp" in data


def test_status_returns_503_when_not_started(client):
    resp = client.get("/api/status")
    assert resp.status_code == 503


def test_submit_requires_text(client):
    resp = client.post("/api/submit", json={})
    assert resp.status_code == 400
    assert "text is required" in resp.json()["error"]


def test_claims_returns_503_when_not_started(client):
    resp = client.get("/api/claims")
    assert resp.status_code == 503


def test_dashboard_serves_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
