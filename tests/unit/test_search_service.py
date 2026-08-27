from app.services import product_service, search_service


def _payload(sku: str) -> dict:
    return {
        "name": f"Produto {sku}",
        "sku": sku,
        "sale_type": "eletronico",
        "brand": "Marca",
        "category": "categoria",
        "description": f"Descricao {sku}",
    }


async def test_search_semantic_returns_empty_when_vector_db_finds_nothing(mongo_database, monkeypatch):
    async def fake_vector_search(embedding, n_results=10, where=None) -> dict:
        return {"ids": [], "distances": [], "metadatas": [], "documents": []}

    from app.clients import vector_db_client

    monkeypatch.setattr(vector_db_client, "search", fake_vector_search)

    results = await search_service.search_semantic("produto qualquer")
    assert results == []


async def test_search_semantic_reranks_and_hydrates_from_mongo(monkeypatch, mongo_database):
    from app.schemas.product import ProductCreate

    product_a = await product_service.create_product(ProductCreate(**_payload("SKU-A")))
    product_b = await product_service.create_product(ProductCreate(**_payload("SKU-B")))

    from app.clients import embedding_reranking_client, vector_db_client

    async def fake_vector_search(embedding, n_results=10, where=None) -> dict:
        # vector-db's naive distance ranking puts A first...
        return {
            "ids": [product_a.id, product_b.id],
            "distances": [0.1, 0.2],
            "metadatas": [{}, {}],
            "documents": ["doc a", "doc b"],
        }

    async def fake_rerank(query: str, passages: list) -> list:
        # ...but reranking flips the order: passages[1] ("doc b") scores higher.
        return [
            {"passage": passages[1], "score": 0.95, "index": 1},
            {"passage": passages[0], "score": 0.40, "index": 0},
        ]

    monkeypatch.setattr(vector_db_client, "search", fake_vector_search)
    monkeypatch.setattr(embedding_reranking_client, "rerank", fake_rerank)

    results = await search_service.search_semantic("produto qualquer")

    assert [doc["sku"] for doc, _ in results] == ["SKU-B", "SKU-A"]
    assert [score for _, score in results] == [0.95, 0.40]
