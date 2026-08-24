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
