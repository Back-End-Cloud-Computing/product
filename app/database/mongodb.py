import logging

from pymongo import AsyncMongoClient
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class MongoDB:
    """Holds the lazily-initialized Mongo client/database for the app lifespan.

    Uses PyMongo's native async API (`pymongo.AsyncMongoClient`, stable since
    PyMongo 4.9) rather than Motor: MongoDB is deprecating Motor in favor of
    this driver-native async support, so there is no reason to take the extra
    dependency for a new service.
    """

    client: AsyncMongoClient | None = None
    database: AsyncDatabase | None = None


mongodb = MongoDB()


async def connect_to_mongo() -> None:
    settings = get_settings()
    mongodb.client = AsyncMongoClient(settings.mongodb_uri)
    mongodb.database = mongodb.client[settings.mongodb_database]

    await mongodb.database.products.create_index("sku", unique=True)
    await mongodb.database.products.create_index(
        [("name", "text"), ("description", "text"), ("brand", "text"), ("category", "text")],
        name="product_text_index",
        default_language="portuguese",
    )
    logger.info("Connected to MongoDB at %s (db=%s)", settings.mongodb_uri, settings.mongodb_database)


async def close_mongo_connection() -> None:
    if mongodb.client is not None:
        await mongodb.client.close()
        logger.info("MongoDB connection closed")


def get_products_collection() -> AsyncCollection:
    if mongodb.database is None:
        raise RuntimeError("MongoDB connection has not been initialized")
    return mongodb.database.products
