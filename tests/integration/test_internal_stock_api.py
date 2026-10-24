import pytest

from app.core.config import get_settings

SECRET = "segredo-de-teste"


@pytest.fixture(autouse=True)
def internal_secret(monkeypatch):
    monkeypatch.setattr(get_settings(), "internal_secret", SECRET)


async def _create_product(api_client, sku: str, stock: int) -> str:
    response = await api_client.post(
        "/products",
        json={
            "name": "P",
            "sku": sku,
            "sale_type": "s",
            "brand": "b",
            "category": "c",
            "price": 10.0,
            "stock": stock,
            "description": "d",
        },
    )
    return response.json()["id"]


async def _stock_call(api_client, product_id, action, quantity, key, secret=SECRET):
    headers = {"Idempotency-Key": key}
    if secret is not None:
        headers["x-internal-secret"] = secret
    return await api_client.post(
        f"/internal/products/{product_id}/stock/{action}", json={"quantity": quantity}, headers=headers
    )


async def _current_stock(api_client, product_id) -> int:
    return (await api_client.get(f"/products/{product_id}")).json()["stock"]


async def test_decrement_returns_remaining_stock(api_client):
    product_id = await _create_product(api_client, "SKU-S1", stock=10)

    response = await _stock_call(api_client, product_id, "decrement", 3, "op-1")

    assert response.status_code == 200
    assert response.json() == {"operation_id": "op-1", "product_id": product_id, "remaining_stock": 7}
    assert await _current_stock(api_client, product_id) == 7


async def test_repeated_idempotency_key_applies_once(api_client):
    product_id = await _create_product(api_client, "SKU-S2", stock=10)

    first = await _stock_call(api_client, product_id, "decrement", 3, "op-2")
    retry = await _stock_call(api_client, product_id, "decrement", 3, "op-2")

    assert retry.status_code == 200
    assert retry.json() == first.json()
    assert await _current_stock(api_client, product_id) == 7


async def test_reused_idempotency_key_with_different_request_is_a_conflict(api_client):
    product_id = await _create_product(api_client, "SKU-S3", stock=10)
    await _stock_call(api_client, product_id, "decrement", 3, "op-3")

    response = await _stock_call(api_client, product_id, "decrement", 5, "op-3")

    assert response.status_code == 409
    assert response.json()["error_type"] == "operation_conflict"
    assert await _current_stock(api_client, product_id) == 7


async def test_insufficient_stock_leaves_stock_untouched_and_key_reusable(api_client):
    product_id = await _create_product(api_client, "SKU-S4", stock=2)

    response = await _stock_call(api_client, product_id, "decrement", 3, "op-4")

    assert response.status_code == 409
    assert response.json()["error_type"] == "insufficient_stock"
    assert response.json()["available_stock"] == 2
    assert await _current_stock(api_client, product_id) == 2

    await api_client.patch(f"/products/{product_id}", json={"stock": 5})
    retry = await _stock_call(api_client, product_id, "decrement", 3, "op-4")
    assert retry.status_code == 200
    assert retry.json()["remaining_stock"] == 2


async def test_restore_adds_stock_back(api_client):
    product_id = await _create_product(api_client, "SKU-S5", stock=10)
    await _stock_call(api_client, product_id, "decrement", 4, "op-5")

    response = await _stock_call(api_client, product_id, "restore", 4, "op-5-compensation")

    assert response.status_code == 200
    assert response.json()["remaining_stock"] == 10


async def test_unknown_product_returns_404(api_client):
    response = await _stock_call(api_client, "64b64b64b64b64b64b64b64b", "decrement", 1, "op-6")

    assert response.status_code == 404
    assert response.json()["error_type"] == "product_not_found"


@pytest.mark.parametrize("secret", [None, "segredo-errado"])
async def test_missing_or_wrong_internal_secret_returns_401(api_client, secret):
    product_id = await _create_product(api_client, f"SKU-S7-{secret}", stock=10)

    response = await _stock_call(api_client, product_id, "decrement", 1, "op-7", secret=secret)

    assert response.status_code == 401
    assert await _current_stock(api_client, product_id) == 10


async def test_unconfigured_internal_secret_refuses_every_call(api_client, monkeypatch):
    product_id = await _create_product(api_client, "SKU-S8", stock=10)
    monkeypatch.setattr(get_settings(), "internal_secret", "")

    response = await _stock_call(api_client, product_id, "decrement", 1, "op-8", secret="")

    assert response.status_code == 401


async def test_missing_idempotency_key_or_non_positive_quantity_returns_422(api_client):
    product_id = await _create_product(api_client, "SKU-S9", stock=10)
    url = f"/internal/products/{product_id}/stock/decrement"

    no_key = await api_client.post(url, json={"quantity": 1}, headers={"x-internal-secret": SECRET})
    zero = await _stock_call(api_client, product_id, "decrement", 0, "op-9")

    assert no_key.status_code == 422
    assert zero.status_code == 422
