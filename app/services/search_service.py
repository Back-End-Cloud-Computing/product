import logging
from typing import Any

from bson import ObjectId

from app.clients import embedding_reranking_client
from app.database import mongodb

logger = logging.getLogger(__name__)


async def search_lexical(query: str, limit: int = 10) -> list[tuple[dict[str, Any], float]]:
    """Keyword-based search over MongoDB's text index."""
    collection = mongodb.get_products_collection()
    cursor = (
        collection.find(
            {"$text": {"$search": query}},
            {"score": {"$meta": "textScore"}},
        )
        .sort([("score", {"$meta": "textScore"})])
        .limit(limit)
    )
    results: list[tuple[dict[str, Any], float]] = []
    async for doc in cursor:
        score = doc.pop("score", 0.0)
        results.append((doc, float(score)))
    return results


async def search_semantic(query: str, limit: int = 10) -> list[tuple[dict[str, Any], float]]:
    """Meaning-based search: asks embedding-reranking to embed the query and run
    the KNN search on vector-db, so results can match even without exact
    keyword overlap."""
    results = await embedding_reranking_client.search(query, n_results=limit)
    ids = results.get("ids", [])
    distances = results.get("distances", [])
    if not ids:
        return []

    collection = mongodb.get_products_collection()
    docs_by_id: dict[str, dict[str, Any]] = {}
    async for doc in collection.find({"_id": {"$in": [ObjectId(i) for i in ids]}}):
        docs_by_id[str(doc["_id"])] = doc

    results: list[tuple[dict[str, Any], float]] = []
    for product_id, distance in zip(ids, distances):
        doc = docs_by_id.get(product_id)
        if doc:
            similarity = max(0.0, 1.0 - distance)
            results.append((doc, similarity))
    return results
