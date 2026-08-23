import pytest

from app.core.exceptions import DescriptionNotApprovedError, DuplicateSkuError, ProductNotFoundError
from app.schemas.product import ProductCreate, ProductUpdate
from app.services import product_service


async def test_create_product(mongo_database):
    payload = ProductCreate(
        name="Produto Teste",
        sku="SKU-100",
        sale_type="eletronico",
        brand="MarcaA",
        category="categoria",
        attributes={"cor": "preto"},
    )
    product = await product_service.create_product(payload)

    assert product.sku == "SKU-100"
    assert product.description_status.value == "pending"
    assert product.embedding_status.value == "pending"


async def test_create_product_with_duplicate_sku_raises(mongo_database):
    payload = ProductCreate(name="A", sku="SKU-DUP", sale_type="eletronico", brand="B", category="C")
    await product_service.create_product(payload)

    with pytest.raises(DuplicateSkuError):
        await product_service.create_product(payload)


async def test_get_missing_product_raises_not_found(mongo_database):
    with pytest.raises(ProductNotFoundError):
        await product_service.get_product("64b64b64b64b64b64b64b64b")


async def test_update_product_partial(mongo_database):
    created = await product_service.create_product(
        ProductCreate(name="A", sku="SKU-200", sale_type="eletronico", brand="B", category="C")
    )

    updated = await product_service.update_product(created.id, ProductUpdate(brand="NovaMarca"))

    assert updated.brand == "NovaMarca"
    assert updated.name == "A"


async def test_description_flow_generate_then_confirm(mongo_database):
    created = await product_service.create_product(
        ProductCreate(name="Notebook", sku="SKU-300", sale_type="eletronico", brand="MarcaB", category="notebooks")
    )

    suggested = await product_service.generate_description_suggestion(created.id)
    assert suggested.description_status.value == "suggested"
    assert suggested.suggested_description

    confirmed = await product_service.confirm_description(created.id, "Descricao final editada pelo usuario.")
    assert confirmed.description_status.value == "approved"
    assert confirmed.description == "Descricao final editada pelo usuario."
    # Embedding sync ran as part of confirmation (fake collaborators from conftest).
    assert confirmed.embedding_status.value == "synced"


async def test_retry_embedding_sync_requires_approved_description(mongo_database):
    created = await product_service.create_product(
        ProductCreate(name="A", sku="SKU-400", sale_type="eletronico", brand="B", category="C")
    )

    with pytest.raises(DescriptionNotApprovedError):
        await product_service.retry_embedding_sync(created.id)


async def test_sync_embedding_failure_is_recorded_not_raised(mongo_database, monkeypatch):
    from app.core.exceptions import EmbeddingGenerationError
    from app.services import embedding_service

    async def broken_generate_embedding(text: str):
        raise EmbeddingGenerationError("boom")

    monkeypatch.setattr(embedding_service, "generate_embedding", broken_generate_embedding)

    created = await product_service.create_product(
        ProductCreate(name="A", sku="SKU-500", sale_type="eletronico", brand="B", category="C")
    )
    confirmed = await product_service.confirm_description(created.id, "Descricao qualquer.")

    assert confirmed.embedding_status.value == "failed"
    assert confirmed.embedding_sync_error is not None
