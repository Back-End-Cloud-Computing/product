from datetime import datetime, timezone

from app.services import recommendation_service


async def _insert_product(mongo_database, **overrides) -> str:
    document = {
        "name": "Produto",
        "sku": "SKU-REC",
        "sale_type": "eletronico",
        "brand": "MarcaZ",
        "category": "categoria",
        "attributes": {},
        "description": "descricao aprovada",
        "suggested_description": None,
        "description_status": "approved",
        "embedding_status": "pending",
        "embedding_sync_error": None,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    document.update(overrides)
    result = await mongo_database["products"].insert_one(document)
    return str(result.inserted_id)


async def test_recommend_for_user_falls_back_to_popular_when_no_signals(mongo_database):
    await _insert_product(mongo_database, sku="SKU-POP")

    results = await recommendation_service.recommend_for_user("user-without-history", limit=5)

    assert len(results) == 1
    assert results[0]["reasons"] == ["produto em destaque"]


async def test_recommend_for_user_combines_signals_from_order_history(mongo_database, monkeypatch):
    referenced_id = await _insert_product(mongo_database, sku="SKU-REF")
    candidate_id = await _insert_product(mongo_database, sku="SKU-CANDIDATE")

    async def fake_get_purchased_product_ids(self, user_id):
        return [referenced_id]

    async def fake_recommend_similar_products(product_id, limit=10):
        assert product_id == referenced_id
        from app.services.product_service import document_to_response

        doc = await mongo_database["products"].find_one({"sku": "SKU-CANDIDATE"})
        return [{"product": document_to_response(doc), "score": 0.9, "reasons": ["similaridade de conteudo"]}]

    from app.integrations import order_client

    monkeypatch.setattr(order_client.MockOrderServiceClient, "get_purchased_product_ids", fake_get_purchased_product_ids)
    monkeypatch.setattr(recommendation_service, "recommend_similar_products", fake_recommend_similar_products)

    results = await recommendation_service.recommend_for_user("user-with-history", limit=5)

    assert len(results) == 1
    assert results[0]["product"].id == candidate_id
    assert any("historico de compras" in reason for reason in results[0]["reasons"])
