import logging
from datetime import datetime, timezone
from typing import Any

from pymongo.errors import DuplicateKeyError

from app.clients import embedding_reranking_client, llm_provider_client, vector_db_client
from app.core.exceptions import (
    DuplicateSkuError,
    EmbeddingGenerationError,
    LLMProviderError,
    ProductNotFoundError,
)
from app.core.mongo_utils import to_object_id
from app.database import mongodb
from app.models.product import ProductDocument
from app.schemas.product import ProductCreate, ProductResponse, ProductUpdate

logger = logging.getLogger(__name__)


def build_semantic_text(
    name: str,
    brand: str,
    category: str,
    description: str,
    attributes: dict[str, Any],
) -> str:
    """Combine the semantically relevant product fields into a single text blob
    sent to embedding-reranking, both as the embedding input and as the
    document stored in the vector-db. Lives here (not in embedding-reranking)
    because it depends on the product's own schema."""
    attributes_text = " ".join(f"{key}: {value}" for key, value in attributes.items())
    parts = [name, brand, category, description, attributes_text]
    return " | ".join(part for part in parts if part)


def build_description_prompt(
    name: str,
    brand: str,
    category: str,
    sale_type: str,
    attributes: dict[str, Any],
) -> str:
    attributes_text = ", ".join(f"{key}: {value}" for key, value in attributes.items())
    if not attributes_text:
        attributes_text = "nenhum atributo adicional informado"

    return (
        "Você é um redator de e-commerce. Escreva uma descrição comercial persuasiva e "
        "objetiva, em português, para o produto abaixo. Use no máximo 3 parágrafos curtos "
        "e não invente características que não foram informadas.\n\n"
        f"Nome: {name}\n"
        f"Marca: {brand}\n"
        f"Categoria: {category}\n"
        f"Tipo de venda: {sale_type}\n"
        f"Atributos: {attributes_text}\n"
    )


def build_summary_prompt(
    name: str,
    brand: str,
    category: str,
    description: str,
    attributes: dict[str, Any],
) -> str:
    attributes_text = ", ".join(f"{key}: {value}" for key, value in attributes.items())
    if not attributes_text:
        attributes_text = "nenhum atributo adicional informado"

    return (
        "Resuma o produto abaixo em frases curtas e objetivas, em português, "
        "para uso interno de indexação e busca. Não use marketing, apenas os fatos.\n\n"
        f"Nome: {name}\n"
        f"Marca: {brand}\n"
        f"Categoria: {category}\n"
        f"Descrição: {description}\n"
        f"Atributos: {attributes_text}\n"
    )


def document_to_response(doc: dict[str, Any]) -> ProductResponse:
    return ProductResponse(
        id=str(doc["_id"]),
        name=doc["name"],
        sku=doc["sku"],
        sale_type=doc["sale_type"],
        brand=doc["brand"],
        category=doc["category"],
        attributes=doc.get("attributes", {}),
        description=doc.get("description"),
        summary=doc.get("summary"),
        error=doc.get("error"),
        created_at=doc["created_at"],
        updated_at=doc["updated_at"],
    )


async def create_product(payload: ProductCreate) -> ProductResponse:
    collection = mongodb.get_products_collection()

    # Explicit pre-check keeps the error path testable/deterministic even against
    # backends that don't enforce unique indexes; the DB-level unique index below
    # is the real safety net against races between concurrent requests.
    if await collection.find_one({"sku": payload.sku}):
        raise DuplicateSkuError(payload.sku)

    description = payload.description
    error: str | None = None

    if payload.generate_description_with_ai:
        prompt = build_description_prompt(
            name=payload.name,
            brand=payload.brand,
            category=payload.category,
            sale_type=payload.sale_type,
            attributes=payload.attributes,
        )
        try:
            description = await llm_provider_client.generate_text(prompt)
        except LLMProviderError as exc:
            logger.error("Description generation failed for sku %s: %s", payload.sku, exc)
            error = type(exc).__name__

    document = ProductDocument(
        name=payload.name,
        sku=payload.sku,
        sale_type=payload.sale_type,
        brand=payload.brand,
        category=payload.category,
        attributes=payload.attributes,
        description=description,
        error=error,
    )
    try:
        result = await collection.insert_one(document.to_mongo())
    except DuplicateKeyError as exc:
        raise DuplicateSkuError(payload.sku) from exc

    product_id = str(result.inserted_id)
    logger.info("Product created: id=%s sku=%s", product_id, payload.sku)

    if description and not error:
        await _finish_product_pipeline(product_id, payload, description, collection)

    created = await collection.find_one({"_id": result.inserted_id})
    return document_to_response(created)


async def _finish_product_pipeline(
    product_id: str,
    payload: ProductCreate,
    description: str,
    collection: Any,
) -> None:
    """Post-creation pipeline: generate the summary via LLM, then index the
    product (including its description) into the vector-db. Best-effort: the
    first failure is recorded on `error` and the pipeline stops there; the
    product document created earlier is never rolled back, and there is no
    retry endpoint."""
    summary_prompt = build_summary_prompt(
        name=payload.name,
        brand=payload.brand,
        category=payload.category,
        description=description,
        attributes=payload.attributes,
    )
    try:
        summary = await llm_provider_client.generate_text(summary_prompt)
    except LLMProviderError as exc:
        logger.error("Summary generation failed for product %s: %s", product_id, exc)
        await collection.update_one({"_id": to_object_id(product_id)}, {"$set": {"error": type(exc).__name__}})
        return

    await collection.update_one({"_id": to_object_id(product_id)}, {"$set": {"summary": summary}})

    metadata = {
        "sku": payload.sku,
        "name": payload.name,
        "brand": payload.brand,
        "category": payload.category,
        "sale_type": payload.sale_type,
        "description": description,
        "summary": summary,
    }
    semantic_text = build_semantic_text(
        name=payload.name,
        brand=payload.brand,
        category=payload.category,
        description=description,
        attributes=payload.attributes,
    )
    try:
        await embedding_reranking_client.index_product(product_id, semantic_text, metadata)
    except EmbeddingGenerationError as exc:
        logger.error("Vector indexing failed for product %s: %s", product_id, exc)
        await collection.update_one({"_id": to_object_id(product_id)}, {"$set": {"error": type(exc).__name__}})


async def get_product(product_id: str) -> ProductResponse:
    collection = mongodb.get_products_collection()
    doc = await collection.find_one({"_id": to_object_id(product_id)})
    if not doc:
        raise ProductNotFoundError(product_id)
    return document_to_response(doc)


async def list_products(
    skip: int = 0,
    limit: int = 20,
    category: str | None = None,
    brand: str | None = None,
) -> tuple[list[ProductResponse], int]:
    collection = mongodb.get_products_collection()
    query: dict[str, Any] = {}
    if category:
        query["category"] = category
    if brand:
        query["brand"] = brand

    total = await collection.count_documents(query)
    cursor = collection.find(query).skip(skip).limit(limit).sort("created_at", -1)
    items = [document_to_response(doc) async for doc in cursor]
    return items, total


async def get_products_batch(product_ids: list[str]) -> list[ProductResponse]:
    """Hydrate a batch of product ids, used to enrich the thin `product_id` +
    `score` results returned by vector-db's recommendation endpoints (that
    service has no MongoDB access, so it can't return full product data)."""
    collection = mongodb.get_products_collection()
    object_ids = [to_object_id(pid) for pid in product_ids]
    cursor = collection.find({"_id": {"$in": object_ids}})
    return [document_to_response(doc) async for doc in cursor]


async def update_product(product_id: str, payload: ProductUpdate) -> ProductResponse:
    """Partial update of product metadata (PATCH semantics: unset fields are left untouched)."""
    collection = mongodb.get_products_collection()
    updates = {key: value for key, value in payload.model_dump(exclude_unset=True).items() if value is not None}
    if not updates:
        return await get_product(product_id)

    updates["updated_at"] = datetime.now(timezone.utc)
    updated = await collection.find_one_and_update(
        {"_id": to_object_id(product_id)},
        {"$set": updates},
        return_document=True,
    )
    if not updated:
        raise ProductNotFoundError(product_id)
    return document_to_response(updated)


async def delete_product(product_id: str) -> None:
    """Product-service stays the orchestrator of the product lifecycle: delete
    from MongoDB (the source of truth) first, then remove the corresponding
    vector directly on vector-db. No embedding needs generating to delete one,
    so this is the one call that skips embedding-reranking entirely."""
    collection = mongodb.get_products_collection()
    result = await collection.delete_one({"_id": to_object_id(product_id)})
    if result.deleted_count == 0:
        raise ProductNotFoundError(product_id)
    await vector_db_client.delete_product(product_id)
