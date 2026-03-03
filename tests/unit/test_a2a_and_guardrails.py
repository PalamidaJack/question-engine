from fastapi import FastAPI
from fastapi.testclient import TestClient

from qe.api.a2a import register_a2a_routes
from qe.api.endpoints.guardrails import register_guardrails_routes
from qe.api.endpoints.memory_ops import register_memory_ops_routes

app = FastAPI()
register_a2a_routes(app=app)
register_guardrails_routes(app=app)
register_memory_ops_routes(app=app)

client = TestClient(app)


def test_agent_card():
    r = client.get("/.well-known/agent.json")
    assert r.status_code == 200
    j = r.json()
    assert "name" in j


def test_create_task_and_messages():
    # create a task
    r = client.post("/api/a2a/tasks", json={"description": "test task"})
    assert r.status_code == 200
    j = r.json()
    assert "task_id" in j
    task_id = j["task_id"]

    # post a message
    r2 = client.post(
        f"/api/a2a/tasks/{task_id}/messages",
        json={"sender": "external", "content": {"k": "v"}},
    )
    assert r2.status_code == 200
    assert r2.json().get("status") == "ok"

    # get task
    r3 = client.get(f"/api/a2a/tasks/{task_id}")
    assert r3.status_code == 200

    # messages list
    r4 = client.get(f"/api/a2a/tasks/{task_id}/messages")
    assert r4.status_code == 200
    assert "messages" in r4.json()


def test_guardrails_status_and_configure():
    r = client.get("/api/guardrails/status")
    assert r.status_code == 200
    j = r.json()
    assert "configured" in j

    # configure may be unavailable in test app lifespan; accept 200 or 503
    r2 = client.post("/api/guardrails/configure", json={"enabled": True})
    assert r2.status_code in (200, 503)
