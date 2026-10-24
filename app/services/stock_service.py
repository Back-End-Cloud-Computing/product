import logging
from datetime import datetime, timezone
from typing import Literal

from pymongo import ReturnDocument
from pymongo.errors import DuplicateKeyError

from app.core.exceptions import (
    InsufficientStockError,
    ProductNotFoundError,
    ProductServiceError,
    StockOperationConflictError,
)
from app.core.mongo_utils import to_object_id
from app.database import mongodb
from app.schemas.stock import StockOperationResponse

logger = logging.getLogger(__name__)

StockAction = Literal["decrement", "restore"]


async def apply_stock_operation(
    product_id: str,
    action: StockAction,
    quantity: int,
    operation_id: str,
) -> StockOperationResponse:
    """Decrement or restore a product's stock exactly once per `operation_id`
    (the caller's Idempotency-Key), so the caller can retry a POST safely.

    The operation id is claimed first (unique `_id` in `stock_operations`)
    and only then is the stock changed. A repeated id with the same request
    returns the original result; with a different request, or while the
    first one is still running, it is a conflict. If the change itself fails
    (unknown product, not enough stock) the claim is released, so the caller
    may retry the same key later."""
    operations = mongodb.get_stock_operations_collection()
    request = {"product_id": product_id, "action": action, "quantity": quantity}

    try:
        await operations.insert_one(
            {"_id": operation_id, **request, "remaining_stock": None, "created_at": datetime.now(timezone.utc)}
        )
    except DuplicateKeyError:
        return await _replay(operation_id, request)

    try:
        remaining_stock = await _change_stock(product_id, action, quantity)
    except ProductServiceError:
        await operations.delete_one({"_id": operation_id})
        raise

    await operations.update_one({"_id": operation_id}, {"$set": {"remaining_stock": remaining_stock}})
    logger.info("Stock %s: product=%s quantity=%s operation=%s", action, product_id, quantity, operation_id)
    return StockOperationResponse(operation_id=operation_id, product_id=product_id, remaining_stock=remaining_stock)


async def _replay(operation_id: str, request: dict) -> StockOperationResponse:
    existing = await mongodb.get_stock_operations_collection().find_one({"_id": operation_id})
    if (
        existing is None
        or existing["remaining_stock"] is None
        or any(existing[key] != value for key, value in request.items())
    ):
        raise StockOperationConflictError(operation_id)
    return StockOperationResponse(
        operation_id=operation_id,
        product_id=existing["product_id"],
        remaining_stock=existing["remaining_stock"],
    )


async def _change_stock(product_id: str, action: StockAction, quantity: int) -> int:
    products = mongodb.get_products_collection()
    object_id = to_object_id(product_id)

    if action == "decrement":
        # The `$gte` guard makes check-and-decrement a single atomic update,
        # so concurrent checkouts can never take stock below zero.
        query = {"_id": object_id, "stock": {"$gte": quantity}}
        delta = -quantity
    else:
        query = {"_id": object_id}
        delta = quantity

    updated = await products.find_one_and_update(
        query,
        {"$inc": {"stock": delta}, "$set": {"updated_at": datetime.now(timezone.utc)}},
        return_document=ReturnDocument.AFTER,
    )
    if updated is not None:
        return updated["stock"]

    product = await products.find_one({"_id": object_id})
    if product is None:
        raise ProductNotFoundError(product_id)
    raise InsufficientStockError(product_id, product.get("stock", 0))
