import logging
from datetime import datetime, timezone
from typing import Any

from pymongo.errors import DuplicateKeyError

from app.clients import embedding_reranking_client, llm_provider_client, vector_db_client
from app.core.exceptions import (
    DescriptionNotApprovedError,
    DuplicateSkuError,
    ProductNotFoundError,
)
from app.core.mongo_utils import to_object_id
from app.database import mongodb
from app.models.product import DescriptionStatus, EmbeddingSyncStatus, ProductDocument
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
        suggested_description=doc.get("suggested_description"),
        description_status=doc.get("description_status", DescriptionStatus.PENDING),
        embedding_status=doc.get("embedding_status", EmbeddingSyncStatus.PENDING),
        embedding_sync_error=doc.get("embedding_sync_error"),
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

    document = ProductDocument(
        name=payload.name,
        sku=payload.sku,
        sale_type=payload.sale_type,
        brand=payload.brand,
        category=payload.category,
        attributes=payload.attributes,
    )
    try:
        result = await collection.insert_one(document.to_mongo())
    except DuplicateKeyError as exc:
        raise DuplicateSkuError(payload.sku) from exc

    created = await collection.find_one({"_id": result.inserted_id})
    logger.info("Product created: id=%s sku=%s", result.inserted_id, payload.sku)
    return document_to_response(created)


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


async def generate_description_suggestion(product_id: str) -> ProductResponse:
    """Ask the LLM for a draft description. Nothing is persisted as final content
    here: the suggestion is stored under `suggested_description` until the user
    reviews/edits it and confirms via `confirm_description`."""
    product = await get_product(product_id)
    prompt = build_description_prompt(
        name=product.name,
        brand=product.brand,
        category=product.category,
        sale_type=product.sale_type,
        attributes=product.attributes,
    )
    suggested_text = await llm_provider_client.generate_description(prompt)

    collection = mongodb.get_products_collection()
    updated = await collection.find_one_and_update(
        {"_id": to_object_id(product_id)},
        {
            "$set": {
                "suggested_description": suggested_text,
                "description_status": DescriptionStatus.SUGGESTED,
                "updated_at": datetime.now(timezone.utc),
            }
        },
        return_document=True,
    )
    logger.info("Description suggested for product %s", product_id)
    return document_to_response(updated)


async def confirm_description(product_id: str, final_description: str) -> ProductResponse:
    """Persist the user-approved description via partial update (PATCH), then
    trigger embedding generation + ChromaDB sync as described in the product flow."""
    collection = mongodb.get_products_collection()
    updated = await collection.find_one_and_update(
        {"_id": to_object_id(product_id)},
        {
            "$set": {
                "description": final_description,
                "description_status": DescriptionStatus.APPROVED,
                "updated_at": datetime.now(timezone.utc),
            }
        },
        return_document=True,
    )
    if not updated:
        raise ProductNotFoundError(product_id)

    response = document_to_response(updated)
    await sync_embedding(response)
    return await get_product(product_id)


async def sync_embedding(product: ProductResponse) -> None:
    """Trigger embedding-reranking to (re)generate the embedding and index it
    into the vector-db.

    MongoDB is the source of truth and has already been updated by the caller
    before this runs: a failure here must never roll back or fail the parent
    request. Instead, the failure is recorded on `embedding_status`/
    `embedding_sync_error` so it is visible to clients and can be retried via
    `retry_embedding_sync`, giving eventual consistency between the two stores.
    """
    collection = mongodb.get_products_collection()
    try:
        text = build_semantic_text(
            name=product.name,
            brand=product.brand,
            category=product.category,
            description=product.description or "",
            attributes=product.attributes,
        )
        await embedding_reranking_client.index_product(
            product_id=product.id,
            text=text,
            metadata={
                "sku": product.sku,
                "brand": product.brand,
                "category": product.category,
                "sale_type": product.sale_type,
            },
        )
        await collection.update_one(
            {"_id": to_object_id(product.id)},
            {"$set": {"embedding_status": EmbeddingSyncStatus.SYNCED, "embedding_sync_error": None}},
        )
    except Exception as exc:  # noqa: BLE001 - any failure degrades to a recorded sync error
        logger.error("Embedding sync failed for product %s: %s", product.id, exc)
        await collection.update_one(
            {"_id": to_object_id(product.id)},
            {"$set": {"embedding_status": EmbeddingSyncStatus.FAILED, "embedding_sync_error": str(exc)}},
        )


async def retry_embedding_sync(product_id: str) -> ProductResponse:
    product = await get_product(product_id)
    if product.description_status != DescriptionStatus.APPROVED:
        raise DescriptionNotApprovedError(product_id)
    await sync_embedding(product)
    return await get_product(product_id)
