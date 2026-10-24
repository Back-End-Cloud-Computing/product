async def test_create_and_get_product(api_client):
    payload = {
        "name": "Smartphone X",
        "sku": "SKU-001",
        "sale_type": "eletronico",
        "brand": "MarcaA",
        "category": "celulares",
        "price": 10.0,
        "attributes": {"cor": "preto", "memoria": "128GB"},
        "description": "Smartphone com tela grande.",
    }

    create_response = await api_client.post("/products", json=payload)
    assert create_response.status_code == 201
    body = create_response.json()
    assert body["sku"] == "SKU-001"
    assert body["description"] == "Smartphone com tela grande."
    # Summary and indexing run as a background task after the 201 is sent.
    assert body["summary"] is None

    get_response = await api_client.get(f"/products/{body['id']}")
    assert get_response.status_code == 200
    stored = get_response.json()
    assert stored["name"] == "Smartphone X"
    assert stored["summary"]
    assert stored["error"] is None


async def test_create_product_with_ai_description(api_client):
    payload = {
        "name": "Notebook Y",
        "sku": "SKU-004",
        "sale_type": "eletronico",
        "brand": "MarcaB",
        "category": "notebooks",
        "price": 10.0,
        "attributes": {"ram": "16GB"},
        "generate_description_with_ai": True,
    }

    response = await api_client.post("/products", json=payload)
    assert response.status_code == 201
    assert response.json()["description"] is None

    stored = (await api_client.get(f"/products/{response.json()['id']}")).json()
    assert stored["description"]
    assert stored["summary"]
    assert stored["error"] is None


async def test_create_product_without_description_or_ai_flag_returns_422(api_client):
    payload = {
        "name": "Produto",
        "sku": "SKU-005",
        "sale_type": "eletronico",
        "brand": "Marca",
        "category": "categoria",
        "price": 10.0,
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
        "price": 10.0,
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
        "price": 10.0,
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
            "price": 10.0,
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
            "price": 10.0,
            "description": "D",
        },
    )
    product_id = create_response.json()["id"]

    delete_response = await api_client.delete(f"/products/{product_id}")
    assert delete_response.status_code == 204

    get_response = await api_client.get(f"/products/{product_id}")
    assert get_response.status_code == 404


async def test_create_product_responds_before_pipeline_runs(api_client, monkeypatch):
    """Regression for the gateway 504: the LLM/indexing steps must not hold the
    POST response. Drives the ASGI app directly to observe the order of events:
    the 201 headers must go out before the first LLM call starts."""
    import asyncio

    from app.clients import llm_provider_client

    pipeline_started = asyncio.Event()

    async def recording_generate_text(prompt: str) -> str:
        pipeline_started.set()
        return "Resumo."

    monkeypatch.setattr(llm_provider_client, "generate_text", recording_generate_text)

    sent_status: list[int] = []
    from app.main import app

    async def receive():
        body = b'{"name": "P", "sku": "SKU-SLOW", "sale_type": "s", "brand": "b", "category": "c", "price": 1.0, "description": "d"}'
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            assert not pipeline_started.is_set()
            sent_status.append(message["status"])

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/products",
        "raw_path": b"/products",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json"), (b"host", b"test")],
        "server": ("test", 80),
        "client": ("test", 1234),
    }
    await app(scope, receive, send)

    assert sent_status == [201]
    assert pipeline_started.is_set()


async def test_product_exposes_price_and_stock(api_client):
    """shopping-cart reads `price` and `stock` straight from GET /products/{id}."""
    create_response = await api_client.post(
        "/products",
        json={
            "name": "Camiseta",
            "sku": "SKU-PRICE",
            "sale_type": "vestuario",
            "brand": "GANJJ",
            "category": "camisetas",
            "price": 49.9,
            "stock": 7,
            "description": "Camiseta de algodao.",
        },
    )
    assert create_response.status_code == 201

    body = (await api_client.get(f"/products/{create_response.json()['id']}")).json()
    assert body["price"] == 49.9
    assert body["stock"] == 7


async def test_create_product_without_price_returns_422(api_client):
    response = await api_client.post(
        "/products",
        json={"name": "P", "sku": "SKU-NOPRICE", "sale_type": "s", "brand": "b", "category": "c", "description": "d"},
    )
    assert response.status_code == 422


async def test_update_product_price_and_stock(api_client):
    create_response = await api_client.post(
        "/products",
        json={
            "name": "P",
            "sku": "SKU-RESTOCK",
            "sale_type": "s",
            "brand": "b",
            "category": "c",
            "price": 10.0,
            "description": "d",
        },
    )
    assert create_response.json()["stock"] == 0

    response = await api_client.patch(
        f"/products/{create_response.json()['id']}", json={"price": 12.5, "stock": 30}
    )

    assert response.status_code == 200
    assert response.json()["price"] == 12.5
    assert response.json()["stock"] == 30
