import logging
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.exceptions import EmbeddingGenerationError

logger = logging.getLogger(__name__)
PRODUCTS_COLLECTION = "products"


async def index_product(product_id: str, text: str, metadata: dict[str, Any]) -> None:
    """Ask embedding-reranking to embed the product's semantic text and index it
    into the vector-db, under the products collection. Raises on failure so the
    caller can record it on the product's `error` field."""
    settings = get_settings()
    payload = {
        "collection_name": PRODUCTS_COLLECTION,
        "id": product_id,
        "text": text,
        "metadata": metadata,
    }
    try:
        async with httpx.AsyncClient(timeout=settings.embedding_reranking_timeout_seconds) as client:
            response = await client.post(f"{settings.embedding_reranking_base_url}/index", json=payload)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.error("embedding-reranking indexing failed for product %s: %s", product_id, exc)
        raise EmbeddingGenerationError(f"Failed to index product '{product_id}': {exc}") from exc


async def embed_query(query: str) -> list[float]:
    """Embeds a single query text via the pure /embed endpoint. Used to run a
    semantic search: product-service embeds the query itself and searches
    vector-db directly, rather than going through an embedding-reranking
    search proxy."""
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=settings.embedding_reranking_timeout_seconds) as client:
            response = await client.post(f"{settings.embedding_reranking_base_url}/embed", json={"texts": [query]})
            response.raise_for_status()
            return response.json()["embeddings"][0]
    except (httpx.HTTPError, KeyError, IndexError) as exc:
        logger.error("embedding-reranking embed failed for query '%s': %s", query, exc)
        raise EmbeddingGenerationError(f"Failed to embed query: {exc}") from exc


async def rerank(query: str, passages: list[str]) -> list[dict[str, Any]]:
    """Reorders `passages` by relevance to `query`. Returns the raw `results`
    list from embedding-reranking (`passage`, `score`, `index` into `passages`)."""
    settings = get_settings()
    payload = {"query": query, "passages": passages}
    try:
        async with httpx.AsyncClient(timeout=settings.embedding_reranking_timeout_seconds) as client:
            response = await client.post(f"{settings.embedding_reranking_base_url}/rerank", json=payload)
            response.raise_for_status()
            return response.json()["results"]
    except (httpx.HTTPError, KeyError) as exc:
        logger.error("embedding-reranking rerank failed for query '%s': %s", query, exc)
        raise EmbeddingGenerationError(f"Failed to rerank search results: {exc}") from exc
