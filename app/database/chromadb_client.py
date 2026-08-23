import asyncio
import logging
from typing import Any

import chromadb

from app.core.config import get_settings
from app.core.exceptions import VectorStoreError

logger = logging.getLogger(__name__)


class ChromaDBState:
    """Holds the lazily-initialized Chroma client/collection for the app lifespan.

    Typed as `Any` deliberately: chromadb's client/collection classes live under
    internal, version-sensitive module paths, so depending on them directly here
    would couple this module to chromadb's implementation details rather than
    its public `chromadb.HttpClient(...)` entry point used below.
    """

    client: Any | None = None
    collection: Any | None = None


chromadb_state = ChromaDBState()


def connect_to_chromadb() -> None:
    settings = get_settings()
    chromadb_state.client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    chromadb_state.collection = chromadb_state.client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("Connected to ChromaDB at %s:%s", settings.chroma_host, settings.chroma_port)


def get_products_collection() -> Any:
    if chromadb_state.collection is None:
        raise RuntimeError("ChromaDB connection has not been initialized")
    return chromadb_state.collection


async def upsert_embedding(
    product_id: str,
    embedding: list[float],
    metadata: dict[str, Any],
    document: str,
) -> None:
    """Upsert (insert or replace) a product's embedding, keyed by its MongoDB id."""

    def _upsert() -> None:
        try:
            get_products_collection().upsert(
                ids=[product_id],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[document],
            )
        except Exception as exc:  # noqa: BLE001 - normalize any backend failure
            raise VectorStoreError(f"Failed to upsert embedding for product '{product_id}': {exc}") from exc

    await asyncio.to_thread(_upsert)


async def query_similar(
    embedding: list[float],
    n_results: int = 10,
    where: dict[str, Any] | None = None,
) -> dict[str, Any]:
    def _query() -> dict[str, Any]:
        try:
            return get_products_collection().query(
                query_embeddings=[embedding],
                n_results=n_results,
                where=where,
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to query ChromaDB: {exc}") from exc

    return await asyncio.to_thread(_query)


async def delete_embedding(product_id: str) -> None:
    """Best-effort delete: a missing/failed removal must not block product deletion."""

    def _delete() -> None:
        try:
            get_products_collection().delete(ids=[product_id])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to delete embedding for product '%s': %s", product_id, exc)

    await asyncio.to_thread(_delete)
