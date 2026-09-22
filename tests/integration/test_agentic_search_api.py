# Like /search/lexical (see test_search_api.py), the agentic search endpoints
# go through search_service.search_lexical/search_semantic directly here
# rather than through mongomock, since mongomock does not implement the $text
# operator used by lexical search.

import json
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.core.security import get_current_user
from app.services import search_service
from tests.conftest import FAKE_USER


def _make_doc(doc_id: str, name: str) -> dict:
    return {
        "_id": doc_id,
        "name": name,
        "sku": f"SKU-{doc_id}",
        "sale_type": "eletronico",
        "brand": "Marca",
        "category": "categoria",
        "attributes": {},
        "description": None,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }


def _patch_search(monkeypatch, lexical_docs=None, semantic_docs=None):
    async def fake_lexical(query, limit=10):
        return [(doc, 1.0) for doc in (lexical_docs or [])]

    async def fake_semantic(query, limit=10):
        return [(doc, 0.9) for doc in (semantic_docs or [])]

    monkeypatch.setattr(search_service, "search_lexical", fake_lexical)
    monkeypatch.setattr(search_service, "search_semantic", fake_semantic)


async def test_agentic_search_post_returns_relevant_documents_with_source_and_reason(api_client, monkeypatch):
    doc = _make_doc("a", "Produto Teste")
    _patch_search(monkeypatch, lexical_docs=[doc], semantic_docs=[doc])

    response = await api_client.post("/search/agentic_search", json={"query": "produto teste", "limit": 5})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "produto teste"
    assert body["total"] == 1
    assert body["iterations"] >= 1

    item = body["results"][0]
    assert item["product"]["id"] == "a"
    # Found by both fakes above, so the aggregation step must label it "both".
    assert item["source"] == "both"
    assert item["reason"]


async def test_agentic_search_requires_non_empty_query(api_client):
    response = await api_client.post("/search/agentic_search", json={"query": ""})

    assert response.status_code == 422


async def test_agentic_search_ws_streams_progress_then_completed(mongo_database, monkeypatch):
    doc = _make_doc("a", "Produto Teste")
    _patch_search(monkeypatch, lexical_docs=[doc], semantic_docs=[])

    from app.main import app

    app.dependency_overrides[get_current_user] = lambda: FAKE_USER
    try:
        client = TestClient(app)
        with client.websocket_connect("/search/agentic_search/ws") as websocket:
            websocket.send_text(json.dumps({"query": "produto teste", "limit": 5}))

            events = [websocket.receive_json()]
            while events[-1]["type"] not in ("completed", "error"):
                events.append(websocket.receive_json())
    finally:
        app.dependency_overrides.pop(get_current_user, None)

    assert events[0]["type"] == "iteration_start"
    assert any(event["type"] == "iteration_result" for event in events)

    final = events[-1]
    assert final["type"] == "completed"
    assert final["results"][0]["product"]["id"] == "a"
    assert final["results"][0]["source"] == "lexical"
