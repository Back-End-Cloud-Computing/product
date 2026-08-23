from app.services import search_service


async def test_hybrid_search_ranks_items_present_in_both_lists_first(monkeypatch):
    doc_a = {"_id": "a", "name": "Produto A"}
    doc_b = {"_id": "b", "name": "Produto B"}
    doc_c = {"_id": "c", "name": "Produto C"}

    async def fake_lexical(query, limit=10):
        return [(doc_a, 2.0), (doc_b, 1.0)]

    async def fake_semantic(query, limit=10):
        return [(doc_b, 0.9), (doc_c, 0.8)]

    monkeypatch.setattr(search_service, "search_lexical", fake_lexical)
    monkeypatch.setattr(search_service, "search_semantic", fake_semantic)

    results = await search_service.search_hybrid("consulta", limit=3)
    ranked_ids = [doc["_id"] for doc, _score in results]

    # "b" appears in both the lexical and semantic rankings, so RRF must place it first.
    assert ranked_ids[0] == "b"
    assert set(ranked_ids) == {"a", "b", "c"}


async def test_hybrid_search_returns_empty_when_no_signal_matches(monkeypatch):
    async def empty_lexical(query, limit=10):
        return []

    async def empty_semantic(query, limit=10):
        return []

    monkeypatch.setattr(search_service, "search_lexical", empty_lexical)
    monkeypatch.setattr(search_service, "search_semantic", empty_semantic)

    results = await search_service.search_hybrid("consulta sem resultados", limit=3)

    assert results == []
