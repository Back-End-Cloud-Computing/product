import logging
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.exceptions import EmbeddingGenerationError

logger = logging.getLogger(__name__)


async def index_product(product_id: str, text: str, metadata: dict[str, Any]) -> None:
    """Ask embedding-reranking to embed the product's semantic text and index it
    into the vector-db. Raises on failure so the caller can record it on
    `embedding_status`/`embedding_sync_error` instead of silently diverging."""
    settings = get_settings()
    payload = {"product_id": product_id, "text": text, "metadata": metadata}
    try:
        async with httpx.AsyncClient(timeout=settings.embedding_reranking_timeout_seconds) as client:
            response = await client.post(f"{settings.embedding_reranking_base_url}/embed/index", json=payload)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.error("embedding-reranking indexing failed for product %s: %s", product_id, exc)
        raise EmbeddingGenerationError(f"Failed to index product '{product_id}': {exc}") from exc


async def search(query: str, n_results: int = 10, where: dict[str, Any] | None = None) -> dict[str, Any]:
    """Ask embedding-reranking to embed the query and run the semantic search
    against the vector-db, returning matched product ids with their scores."""
    settings = get_settings()
    payload: dict[str, Any] = {"query": query, "n_results": n_results, "where": where}
    try:
        async with httpx.AsyncClient(timeout=settings.embedding_reranking_timeout_seconds) as client:
            response = await client.post(f"{settings.embedding_reranking_base_url}/search", json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:
        logger.error("embedding-reranking search failed for query '%s': %s", query, exc)
        raise EmbeddingGenerationError(f"Failed to run semantic search: {exc}") from exc
