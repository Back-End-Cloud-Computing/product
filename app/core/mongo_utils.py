from bson import ObjectId
from bson.errors import InvalidId

from app.core.exceptions import ProductNotFoundError


def to_object_id(product_id: str) -> ObjectId:
    """Parse a Mongo ObjectId, translating malformed ids into a domain 404."""
    try:
        return ObjectId(product_id)
    except InvalidId as exc:
        raise ProductNotFoundError(product_id) from exc
