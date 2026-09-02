import logging
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.exceptions import EmbeddingGenerationError
from app.core.http_retry import post_with_retry

logger = logging.getLogger(__name__)

PRODUCTS_COLLECTION = "products"


async def search(embedding: list[float], n_results: int = 10, where: dict[str, Any] | None = None) -> dict[str, Any]:
    """Runs the KNN search directly on vector-db, within the products
    collection. product-service embeds the query itself (via
    embedding_reranking_client.embed_query) rather than going through a
    search proxy on embedding-reranking - that service only owns the *write*
    path (embedding + indexing) into vector-db."""
    settings = get_settings()
    payload: dict[str, Any] = {
        "collection_name": PRODUCTS_COLLECTION,
        "embedding": embedding,
        "n_results": n_results,
        "where": where,
    }
    try:
        async with httpx.AsyncClient(timeout=settings.vector_db_timeout_seconds) as client:
            response = await post_with_retry(client, f"{settings.vector_db_base_url}/vector_db/search", payload)
            return response.json()
    except httpx.HTTPError as exc:
        logger.error("vector-db search failed: %s", exc)
        raise EmbeddingGenerationError(f"Failed to run semantic search: {exc}") from exc


async def delete_product(product_id: str) -> None:
    """Best-effort delete of the product's vector record, called directly on
    the vector-db (the only direct product-service -> vector-db call: there is
    no embedding to generate for a deletion, so routing it through
    embedding-reranking would be pointless indirection). A missing/failed
    removal must not block product deletion from MongoDB, which has already
    happened by the time this runs."""
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=settings.vector_db_timeout_seconds) as client:
            await post_with_retry(
                client,
                f"{settings.vector_db_base_url}/vector_db/delete",
                {"collection_name": PRODUCTS_COLLECTION, "ids": [product_id]},
            )
    except httpx.HTTPError as exc:
        logger.warning("Failed to delete vector for product '%s': %s", product_id, exc)
