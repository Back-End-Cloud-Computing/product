import logging
from typing import Any

from bson import ObjectId

from app.clients import embedding_reranking_client, vector_db_client
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
    """Meaning-based search: embeds the query, runs the KNN search directly on
    vector-db, then reranks the candidates via embedding-reranking so results
    can match even without exact keyword overlap and come back in a more
    relevant order than raw vector distance alone."""
    embedding = await embedding_reranking_client.embed_query(query)
    vector_result = await vector_db_client.search(embedding, n_results=limit)
    ids = vector_result.get("ids", [])
    documents = vector_result.get("documents", [])
    if not ids:
        return []

    ranked = await embedding_reranking_client.rerank(query, documents)
    ordered_ids_scores = [(ids[item["index"]], item["score"]) for item in ranked]

    collection = mongodb.get_products_collection()
    docs_by_id: dict[str, dict[str, Any]] = {}
    async for doc in collection.find({"_id": {"$in": [ObjectId(i) for i in ids]}}):
        docs_by_id[str(doc["_id"])] = doc

    results: list[tuple[dict[str, Any], float]] = []
    for product_id, score in ordered_ids_scores:
        doc = docs_by_id.get(product_id)
        if doc:
            results.append((doc, score))
    return results
