import logging
from typing import Any

from bson import ObjectId

from app.core.exceptions import ProductNotFoundError
from app.core.mongo_utils import to_object_id
from app.database import chromadb_client, mongodb
from app.integrations import get_cart_client, get_customer_client, get_order_client
from app.services import embedding_service
from app.services.product_service import document_to_response

logger = logging.getLogger(__name__)

# Behavioral signals are weighted rather than relying purely on vector
# similarity, so purchase/view/cart history can shift recommendations even
# when content similarity alone would rank items differently.
WEIGHT_PURCHASE_HISTORY = 0.35
WEIGHT_VIEWED_PRODUCTS = 0.25
WEIGHT_CART = 0.25
WEIGHT_SIMILAR_USERS = 0.15


async def recommend_similar_products(product_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """Content-based recommendation: nearest neighbors of a product's own embedding."""
    collection = mongodb.get_products_collection()
    doc = await collection.find_one({"_id": to_object_id(product_id)})
    if not doc:
        raise ProductNotFoundError(product_id)

    product = document_to_response(doc)
    text = embedding_service.build_semantic_text(
        name=product.name,
        brand=product.brand,
        category=product.category,
        description=product.description or "",
        attributes=product.attributes,
    )
    embedding = await embedding_service.generate_embedding(text)
    chroma_results = await chromadb_client.query_similar(embedding, n_results=limit + 1)

    ids = chroma_results.get("ids", [[]])[0]
    distances = chroma_results.get("distances", [[]])[0]

    other_ids = [i for i in ids if i != product_id]
    docs_by_id: dict[str, dict[str, Any]] = {}
    if other_ids:
        async for d in collection.find({"_id": {"$in": [ObjectId(i) for i in other_ids]}}):
            docs_by_id[str(d["_id"])] = d

    results: list[dict[str, Any]] = []
    for pid, distance in zip(ids, distances):
        if pid == product_id or pid not in docs_by_id:
            continue
        similarity = max(0.0, 1.0 - distance)
        results.append(
            {
                "product": document_to_response(docs_by_id[pid]),
                "score": round(similarity, 4),
                "reasons": ["similaridade de conteudo com o produto de referencia"],
            }
        )
        if len(results) >= limit:
            break
    return results


async def recommend_for_user(user_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """Personalized recommendation combining multiple behavioral signals with
    content similarity. Cart/Order/Customer data is mocked today (those
    microservices aren't integrated yet) behind `app.integrations`, so swapping
    in real HTTP clients later requires no change here.
    """
    cart_ids = await get_cart_client().get_cart_product_ids(user_id)
    purchased_ids = await get_order_client().get_purchased_product_ids(user_id)
    viewed_ids = await get_customer_client().get_viewed_product_ids(user_id)
    similar_users_ids = await get_customer_client().get_similar_users_purchases(user_id)

    weighted_signals = {
        "historico de compras": (purchased_ids, WEIGHT_PURCHASE_HISTORY),
        "produtos visualizados": (viewed_ids, WEIGHT_VIEWED_PRODUCTS),
        "carrinho atual": (cart_ids, WEIGHT_CART),
        "usuarios com comportamento similar": (similar_users_ids, WEIGHT_SIMILAR_USERS),
    }

    scores: dict[str, float] = {}
    reasons: dict[str, list[str]] = {}

    for label, (reference_ids, weight) in weighted_signals.items():
        for reference_id in reference_ids:
            try:
                neighbors = await recommend_similar_products(reference_id, limit=5)
            except ProductNotFoundError:
                continue
            for item in neighbors:
                pid = item["product"].id
                scores[pid] = scores.get(pid, 0.0) + item["score"] * weight
                reasons.setdefault(pid, []).append(f"relacionado a produto de {label}")

    already_owned_or_in_cart = set(cart_ids) | set(purchased_ids)
    ranked = sorted(
        ((pid, score) for pid, score in scores.items() if pid not in already_owned_or_in_cart),
        key=lambda item: item[1],
        reverse=True,
    )[:limit]

    if not ranked:
        return await _fallback_popular_products(limit)

    collection = mongodb.get_products_collection()
    docs: dict[str, dict[str, Any]] = {}
    async for d in collection.find({"_id": {"$in": [ObjectId(pid) for pid, _ in ranked]}}):
        docs[str(d["_id"])] = d

    return [
        {
            "product": document_to_response(docs[pid]),
            "score": round(score, 4),
            "reasons": reasons.get(pid, []),
        }
        for pid, score in ranked
        if pid in docs
    ]


async def _fallback_popular_products(limit: int) -> list[dict[str, Any]]:
    """Cold-start fallback: with no behavioral signal at all (new/anonymous
    user, mocks returning empty data), surface recently approved products
    instead of an empty response."""
    collection = mongodb.get_products_collection()
    cursor = collection.find({"description_status": "approved"}).sort("created_at", -1).limit(limit)
    return [
        {"product": document_to_response(doc), "score": 0.0, "reasons": ["produto em destaque"]}
        async for doc in cursor
    ]
