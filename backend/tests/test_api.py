"""HTTP API endpoints against the real app (temp SQLite, light models, demo corpus)."""
import time

import pytest
from fastapi.testclient import TestClient

QUESTION = "Do vision transformers need large-scale pre-training to outperform convolutional networks?"


@pytest.fixture
def client(api_env):
    from app.main import create_app

    with TestClient(create_app()) as c:
        yield c


def wait_for(client, analysis_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/v1/research/{analysis_id}?include_result=false").json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.2)
    raise AssertionError("analysis did not finish")


def test_health(client):
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok" and body["demo_mode"] is True
    assert set(body["keys_configured"]) >= {"openalex_api_key", "semantic_scholar_api_key"}


def test_validation_errors(client):
    assert client.post("/api/v1/research/analyze", json={"question": "hi"}).status_code == 422
    assert client.post("/api/v1/research/analyze", json={"question": QUESTION, "max_papers": 99}).status_code == 422


def test_unknown_analysis_404(client):
    assert client.get("/api/v1/research/doesnotexist").status_code == 404
    assert client.get("/api/v1/research/doesnotexist/claims").status_code == 404


def test_full_analysis_flow(client):
    r = client.post("/api/v1/research/analyze", json={"question": QUESTION, "max_papers": 6})
    assert r.status_code == 202
    summary = r.json()
    assert [s["key"] for s in summary["stages"]] == ["search", "select", "acquire", "extract", "compare", "graph", "landscape"]

    done = wait_for(client, summary["id"])
    assert done["status"] == "completed", done["error"]
    assert all(s["status"] == "done" for s in done["stages"])
    aid = summary["id"]

    papers = client.get(f"/api/v1/research/{aid}/papers").json()
    claims = client.get(f"/api/v1/research/{aid}/claims").json()
    graph = client.get(f"/api/v1/research/{aid}/graph").json()
    conflicts = client.get(f"/api/v1/research/{aid}/conflicts").json()
    gaps = client.get(f"/api/v1/research/{aid}/gaps").json()
    rels = client.get(f"/api/v1/research/{aid}/relationships").json()
    assert len(papers) == 6 and claims and graph["nodes"]
    assert isinstance(conflicts, list) and "gaps" in gaps and isinstance(rels, list)

    one = claims[0]
    filtered = client.get(f"/api/v1/research/{aid}/claims", params={"paper_id": one["paper_id"]}).json()
    assert all(c["paper_id"] == one["paper_id"] for c in filtered)
    detail = client.get(f"/api/v1/research/{aid}/claims/{one['claim_id']}").json()
    assert detail["paper"]["paper_id"] == one["paper_id"]
    assert detail["claim"]["evidence"][0]["text"]

    listing = client.get("/api/v1/research").json()
    assert listing[0]["id"] == aid and listing[0]["counts"]["claims"] == len(claims)
