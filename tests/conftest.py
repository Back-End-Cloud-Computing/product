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
    """Keeps every test offline by faking only the actual I/O boundaries — the
    outbound HTTP/WS calls to embedding-reranking, vector-db and llm-provider —
    NOT the service functions that contain logic under test, so their real
    error-handling/fallback behavior still runs in tests."""
    from app.clients import embedding_reranking_client, llm_provider_client, vector_db_client

    async def fake_index_product(product_id, text, metadata) -> None:
        return None

    async def fake_search(query, n_results=10, where=None) -> dict:
        return {"ids": [], "distances": [], "metadatas": [], "documents": []}

    async def fake_delete_product(product_id) -> None:
        return None

    async def fake_generate_text(prompt: str) -> str:
        return (
            "Descrição gerada automaticamente em modo offline (nenhum provedor de LLM "
            "configurado ou disponível no momento)."
        )

    monkeypatch.setattr(embedding_reranking_client, "index_product", fake_index_product)
    monkeypatch.setattr(embedding_reranking_client, "search", fake_search)
    monkeypatch.setattr(vector_db_client, "delete_product", fake_delete_product)
    monkeypatch.setattr(llm_provider_client, "generate_text", fake_generate_text)
