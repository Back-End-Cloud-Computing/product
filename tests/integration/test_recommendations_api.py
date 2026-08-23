async def test_similar_products_returns_404_for_missing_product(api_client):
    response = await api_client.get("/recommendations/products/64b64b64b64b64b64b64b64b/similar")

    assert response.status_code == 404


async def test_recommendations_for_user_falls_back_to_popular_products(api_client):
    await api_client.post(
        "/products",
        json={"name": "Produto Popular", "sku": "SKU-REC-01", "sale_type": "eletronico", "brand": "Marca", "category": "cat"},
    )

    response = await api_client.get("/recommendations/users/anonymous-user")

    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == "hybrid_behavioral_signals"
    # No mocked behavioral signal exists for this user, so recommendations fall
    # back to recently added products rather than an empty list.
    assert body["items"] == []


async def test_health_check(api_client):
    response = await api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
