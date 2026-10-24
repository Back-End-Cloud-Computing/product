import pytest

from app.core.exceptions import DuplicateSkuError, EmbeddingGenerationError, LLMProviderError, ProductNotFoundError
from app.schemas.product import ProductCreate, ProductUpdate
from app.services import product_service


async def _create_and_run_pipeline(payload: ProductCreate):
    """Mirrors what `POST /products` does: create, then run the post-creation
    pipeline (a background task in the route), then read the final state."""
    created = await product_service.create_product(payload)
    await product_service.run_post_creation_pipeline(created.id, payload)
    return await product_service.get_product(created.id)


async def test_create_product_with_own_description(mongo_database):
    payload = ProductCreate(
        name="Produto Teste",
        sku="SKU-100",
        sale_type="eletronico",
        brand="MarcaA",
        category="categoria",
        attributes={"cor": "preto"},
        description="Descricao fornecida pelo usuario.",
    )
    product = await _create_and_run_pipeline(payload)

    assert product.sku == "SKU-100"
    assert product.description == "Descricao fornecida pelo usuario."
    assert product.summary
    assert product.error is None


async def test_create_product_with_ai_description(mongo_database):
    payload = ProductCreate(
        name="Notebook",
        sku="SKU-101",
        sale_type="eletronico",
        brand="MarcaB",
        category="notebooks",
        generate_description_with_ai=True,
    )
    product = await _create_and_run_pipeline(payload)

    assert product.description
    assert product.summary
    assert product.error is None


async def test_create_product_with_duplicate_sku_raises(mongo_database):
    payload = ProductCreate(
        name="A", sku="SKU-DUP", sale_type="eletronico", brand="B", category="C", description="D"
    )
    await product_service.create_product(payload)

    with pytest.raises(DuplicateSkuError):
        await product_service.create_product(payload)


async def test_get_missing_product_raises_not_found(mongo_database):
    with pytest.raises(ProductNotFoundError):
        await product_service.get_product("64b64b64b64b64b64b64b64b")


async def test_update_product_partial(mongo_database):
    created = await product_service.create_product(
        ProductCreate(name="A", sku="SKU-200", sale_type="eletronico", brand="B", category="C", description="D")
    )

    updated = await product_service.update_product(created.id, ProductUpdate(brand="NovaMarca"))

    assert updated.brand == "NovaMarca"
    assert updated.name == "A"


async def test_ai_description_failure_is_recorded_and_pipeline_stops(mongo_database, monkeypatch):
    from app.clients import llm_provider_client

    async def broken_generate_text(prompt: str) -> str:
        raise LLMProviderError("boom")

    monkeypatch.setattr(llm_provider_client, "generate_text", broken_generate_text)

    created = await _create_and_run_pipeline(
        ProductCreate(
            name="A", sku="SKU-300", sale_type="eletronico", brand="B", category="C", generate_description_with_ai=True
        )
    )

    assert created.description is None
    assert created.summary is None
    assert created.error == "LLMProviderError"


async def test_summary_failure_is_recorded_but_product_and_description_persist(mongo_database, monkeypatch):
    from app.clients import llm_provider_client

    async def broken_generate_text(prompt: str) -> str:
        raise LLMProviderError("boom")

    monkeypatch.setattr(llm_provider_client, "generate_text", broken_generate_text)

    created = await _create_and_run_pipeline(
        ProductCreate(name="A", sku="SKU-400", sale_type="eletronico", brand="B", category="C", description="D")
    )

    assert created.description == "D"
    assert created.summary is None
    assert created.error == "LLMProviderError"


async def test_indexing_failure_is_recorded_not_raised(mongo_database, monkeypatch):
    from app.clients import embedding_reranking_client

    async def broken_index_product(product_id, text, metadata):
        raise EmbeddingGenerationError("boom")

    monkeypatch.setattr(embedding_reranking_client, "index_product", broken_index_product)

    created = await _create_and_run_pipeline(
        ProductCreate(name="A", sku="SKU-500", sale_type="eletronico", brand="B", category="C", description="D")
    )

    assert created.description == "D"
    assert created.summary is not None
    assert created.error == "EmbeddingGenerationError"


async def test_create_product_returns_before_calling_llm_or_indexing(mongo_database, monkeypatch):
    from app.clients import embedding_reranking_client, llm_provider_client

    async def unexpected_call(*args, **kwargs):
        raise AssertionError("create_product must not wait on the slow pipeline")

    monkeypatch.setattr(llm_provider_client, "generate_text", unexpected_call)
    monkeypatch.setattr(embedding_reranking_client, "index_product", unexpected_call)

    created = await product_service.create_product(
        ProductCreate(
            name="A", sku="SKU-600", sale_type="eletronico", brand="B", category="C", generate_description_with_ai=True
        )
    )

    assert created.description is None
    assert created.summary is None
    assert created.error is None
