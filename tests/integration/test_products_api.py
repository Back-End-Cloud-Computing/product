async def test_create_and_get_product(api_client):
    payload = {
        "name": "Smartphone X",
        "sku": "SKU-001",
        "sale_type": "eletronico",
        "brand": "MarcaA",
        "category": "celulares",
        "attributes": {"cor": "preto", "memoria": "128GB"},
        "description": "Smartphone com tela grande.",
    }

    create_response = await api_client.post("/products", json=payload)
    assert create_response.status_code == 201
    body = create_response.json()
    assert body["sku"] == "SKU-001"
    assert body["description"] == "Smartphone com tela grande."
    assert body["summary"]
    assert body["error"] is None

    get_response = await api_client.get(f"/products/{body['id']}")
    assert get_response.status_code == 200
    assert get_response.json()["name"] == "Smartphone X"


async def test_create_product_with_ai_description(api_client):
    payload = {
        "name": "Notebook Y",
        "sku": "SKU-004",
        "sale_type": "eletronico",
        "brand": "MarcaB",
        "category": "notebooks",
        "attributes": {"ram": "16GB"},
        "generate_description_with_ai": True,
    }

    response = await api_client.post("/products", json=payload)
    assert response.status_code == 201
    body = response.json()
    assert body["description"]
    assert body["summary"]
    assert body["error"] is None


async def test_create_product_without_description_or_ai_flag_returns_422(api_client):
    payload = {
        "name": "Produto",
        "sku": "SKU-005",
        "sale_type": "eletronico",
        "brand": "Marca",
        "category": "categoria",
    }
    response = await api_client.post("/products", json=payload)
    assert response.status_code == 422


async def test_create_product_with_both_description_and_ai_flag_returns_422(api_client):
    payload = {
        "name": "Produto",
        "sku": "SKU-006",
        "sale_type": "eletronico",
        "brand": "Marca",
        "category": "categoria",
        "description": "Minha descricao",
        "generate_description_with_ai": True,
    }
    response = await api_client.post("/products", json=payload)
    assert response.status_code == 422


async def test_create_product_with_duplicate_sku_returns_409(api_client):
    payload = {
        "name": "Produto",
        "sku": "SKU-DUP",
        "sale_type": "eletronico",
        "brand": "Marca",
        "category": "categoria",
        "description": "D",
    }

    first = await api_client.post("/products", json=payload)
    assert first.status_code == 201

    second = await api_client.post("/products", json=payload)
    assert second.status_code == 409
    assert second.json()["error_type"] == "duplicate_sku"


async def test_get_nonexistent_product_returns_404(api_client):
    response = await api_client.get("/products/64b64b64b64b64b64b64b64b")
    assert response.status_code == 404
    assert response.json()["error_type"] == "product_not_found"


async def test_update_product_partial(api_client):
    create_response = await api_client.post(
        "/products",
        json={
            "name": "Produto",
            "sku": "SKU-002",
            "sale_type": "eletronico",
            "brand": "Marca",
            "category": "cat",
            "description": "D",
        },
    )
    product_id = create_response.json()["id"]

    update_response = await api_client.patch(f"/products/{product_id}", json={"brand": "NovaMarca"})

    assert update_response.status_code == 200
    assert update_response.json()["brand"] == "NovaMarca"


async def test_delete_product(api_client):
    create_response = await api_client.post(
        "/products",
        json={
            "name": "Produto",
            "sku": "SKU-003",
            "sale_type": "eletronico",
            "brand": "Marca",
            "category": "cat",
            "description": "D",
        },
    )
    product_id = create_response.json()["id"]

    delete_response = await api_client.delete(f"/products/{product_id}")
    assert delete_response.status_code == 204

    get_response = await api_client.get(f"/products/{product_id}")
    assert get_response.status_code == 404
