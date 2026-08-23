import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from mongomock_motor import AsyncMongoMockClient

from app.database import mongodb as mongodb_module


@pytest_asyncio.fixture
async def mongo_database(monkeypatch):
    """Replaces the real MongoDB client with an in-memory mock.

    `mongomock-motor` mimics Motor's async collection/cursor interface, which
    is method-for-method compatible with the one PyMongo's native async API
    (`pymongo.AsyncMongoClient`, used in `app.database.mongodb`) exposes — so
    it remains a valid stand-in here even though production code no longer
    depends on Motor itself.

    The FastAPI `lifespan` (which connects to real Mongo/Chroma) never runs in
    these tests because `httpx.ASGITransport` does not trigger ASGI lifespan
    events by default, so this fixture is what makes `mongodb.database`
    usable for both service-level unit tests and API-level integration tests.
    """
    client = AsyncMongoMockClient()
    database = client["product_service_test"]
    monkeypatch.setattr(mongodb_module.mongodb, "client", client)
    monkeypatch.setattr(mongodb_module.mongodb, "database", database)
    yield database


@pytest_asyncio.fixture
async def api_client(mongo_database):
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture(autouse=True)
def patch_external_dependencies(monkeypatch):
    """Keeps every test offline by faking only the actual I/O boundaries
    (model loading, ChromaDB network calls) — NOT the service functions that
    contain logic under test (`generate_embedding`, `get_llm_provider`), so
    their real error-handling/fallback behavior still runs in tests.

    The LLM path needs no fake at all: `Settings.llm_provider` already
    defaults to "mock", so `llm_service.get_llm_provider()` naturally resolves
    to the offline `MockLLMProvider` unless a test explicitly opts into
    `openrouter` via env vars.
    """
    from app.database import chromadb_client
    from app.services import embedding_service

    class _FakeEmbeddingModel:
        def encode(self, text: str, normalize_embeddings: bool = True):
            import numpy as np

            return np.array([0.1, 0.2, 0.3])

    async def fake_get_embedding_model() -> _FakeEmbeddingModel:
        return _FakeEmbeddingModel()

    async def fake_upsert_embedding(product_id, embedding, metadata, document) -> None:
        return None

    async def fake_query_similar(embedding, n_results=10, where=None) -> dict:
        return {"ids": [[]], "distances": [[]]}

    monkeypatch.setattr(embedding_service, "get_embedding_model", fake_get_embedding_model)
    monkeypatch.setattr(chromadb_client, "upsert_embedding", fake_upsert_embedding)
    monkeypatch.setattr(chromadb_client, "query_similar", fake_query_similar)
