import logging

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)

PRODUCTS_COLLECTION = "products"


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
            response = await client.post(
                f"{settings.vector_db_base_url}/vector_db/delete",
                json={"collection_name": PRODUCTS_COLLECTION, "ids": [product_id]},
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("Failed to delete vector for product '%s': %s", product_id, exc)
