import logging
from typing import Any

from bson import ObjectId

from app.database import chromadb_client, mongodb
from app.services import embedding_service

logger = logging.getLogger(__name__)

# Reciprocal Rank Fusion constant: a higher K flattens the influence of rank
# position, so results with only one strong signal (lexical OR semantic) aren't
# completely dominated by items that merely appear in both lists.
RRF_K = 60


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
    """Meaning-based search: embeds the query and finds nearest product vectors
    in ChromaDB, so results can match even without exact keyword overlap."""
    embedding = await embedding_service.generate_embedding(query)
    chroma_results = await chromadb_client.query_similar(embedding, n_results=limit)

    ids = chroma_results.get("ids", [[]])[0]
    distances = chroma_results.get("distances", [[]])[0]
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


async def search_hybrid(query: str, limit: int = 10) -> list[tuple[dict[str, Any], float]]:
    """Combines lexical and semantic rankings via Reciprocal Rank Fusion (RRF).

    RRF is used instead of a weighted sum of raw scores because lexical
    (textScore) and semantic (cosine similarity) scores live on different,
    backend-specific scales that aren't directly comparable; fusing by rank
    position avoids having to hand-tune a normalization between them.
    """
    lexical_results = await search_lexical(query, limit=limit * 2)
    semantic_results = await search_semantic(query, limit=limit * 2)

    fused_scores: dict[str, float] = {}
    docs_by_id: dict[str, dict[str, Any]] = {}

    for rank, (doc, _score) in enumerate(lexical_results):
        product_id = str(doc["_id"])
        docs_by_id[product_id] = doc
        fused_scores[product_id] = fused_scores.get(product_id, 0.0) + 1.0 / (RRF_K + rank + 1)

    for rank, (doc, _score) in enumerate(semantic_results):
        product_id = str(doc["_id"])
        docs_by_id[product_id] = doc
        fused_scores[product_id] = fused_scores.get(product_id, 0.0) + 1.0 / (RRF_K + rank + 1)

    ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    return [(docs_by_id[product_id], score) for product_id, score in ranked]
