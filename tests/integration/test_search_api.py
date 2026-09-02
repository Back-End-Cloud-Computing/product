# Note: /search/lexical is not exercised end-to-end here because mongomock
# (used to fake MongoDB in tests) does not implement the $text operator. The
# equivalent lexical-strategy path of /search/agentic_search is covered at the
# unit level in tests/unit/test_agentic_search_service.py instead.


async def test_semantic_search_returns_empty_when_no_matches(api_client):
    """With the conftest-level fake ChromaDB client returning no ids, semantic
    search degrades gracefully to an empty result set instead of erroring."""
    response = await api_client.get("/search/semantic", params={"q": "produto qualquer"})

    assert response.status_code == 200
    body = response.json()
    assert body["results"] == []
    assert body["total"] == 0


async def test_search_requires_non_empty_query(api_client):
    response = await api_client.get("/search/lexical", params={"q": ""})

    assert response.status_code == 422


async def test_semantic_search_falls_back_to_lexical_when_embedding_unavailable(api_client, monkeypatch):
    """When embedding-reranking/vector-db are unreachable, /search/semantic must
    return a degraded (lexical) result set with a 200, not a hard failure."""
    from datetime import datetime, timezone

    from app.core.exceptions import EmbeddingGenerationError
    from app.services import search_service

    doc = {
        "_id": "507f1f77bcf86cd799439011",
        "name": "Produto Fallback",
        "sku": "SKU-FALLBACK",
        "sale_type": "eletronico",
        "brand": "Marca",
        "category": "categoria",
        "attributes": {},
        "description": None,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }

    async def failing_semantic(query, limit=10):
        raise EmbeddingGenerationError("embedding-reranking is down")

    async def fake_lexical(query, limit=10):
        return [(doc, 1.0)]

    monkeypatch.setattr(search_service, "search_semantic", failing_semantic)
    monkeypatch.setattr(search_service, "search_lexical", fake_lexical)

    response = await api_client.get("/search/semantic", params={"q": "produto"})

    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["match_type"] == "lexical_fallback"
    assert body["results"][0]["product"]["sku"] == "SKU-FALLBACK"
