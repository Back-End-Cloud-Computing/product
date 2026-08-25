import json
from contextlib import asynccontextmanager

import httpx
import pytest
import respx

from app.clients import embedding_reranking_client, llm_provider_client, vector_db_client
from app.core.config import get_settings
from app.core.exceptions import EmbeddingGenerationError, LLMProviderError


@pytest.fixture(autouse=True)
def patch_external_dependencies():
    """Overrides (shadows) the conftest-level autouse fixture of the same name:
    these tests exercise the real client implementations against a mocked HTTP
    transport (respx) or a fake WebSocket, so they must NOT have their target
    functions replaced with no-op fakes first."""
    yield


async def test_index_product_calls_embedding_reranking_with_expected_payload():
    settings = get_settings()
    with respx.mock(base_url=settings.embedding_reranking_base_url) as mock:
        route = mock.post("/embed/index").mock(
            return_value=httpx.Response(200, json={"product_id": "1", "status": "indexed", "model": "m"})
        )
        await embedding_reranking_client.index_product("1", "some text", {"brand": "X"})

    assert route.called
    body = json.loads(route.calls.last.request.content)
    assert body == {"product_id": "1", "text": "some text", "metadata": {"brand": "X"}}


async def test_index_product_raises_embedding_generation_error_on_failure():
    settings = get_settings()
    with respx.mock(base_url=settings.embedding_reranking_base_url) as mock:
        mock.post("/embed/index").mock(return_value=httpx.Response(503))
        with pytest.raises(EmbeddingGenerationError):
            await embedding_reranking_client.index_product("1", "text", {})


async def test_search_returns_embedding_reranking_response():
    settings = get_settings()
    with respx.mock(base_url=settings.embedding_reranking_base_url) as mock:
        mock.post("/search").mock(
            return_value=httpx.Response(
                200, json={"ids": ["1"], "distances": [0.1], "metadatas": [{}], "documents": ["doc"]}
            )
        )
        result = await embedding_reranking_client.search("query", n_results=5)

    assert result["ids"] == ["1"]


async def test_vector_db_delete_is_best_effort_and_does_not_raise():
    settings = get_settings()
    with respx.mock(base_url=settings.vector_db_base_url) as mock:
        route = mock.post("/vector_db/delete").mock(return_value=httpx.Response(500))
        await vector_db_client.delete_product("1")  # must not raise

    assert route.called


async def test_llm_provider_generate_text_returns_text():
    settings = get_settings()
    with respx.mock(base_url=settings.llm_provider_base_url) as mock:
        mock.post("/generate").mock(
            return_value=httpx.Response(200, json={"text": "ola", "model": "m", "provider": "mock"})
        )
        result = await llm_provider_client.generate_text("prompt")

    assert result == "ola"


async def test_llm_provider_generate_text_raises_llm_provider_error_on_failure():
    settings = get_settings()
    with respx.mock(base_url=settings.llm_provider_base_url) as mock:
        mock.post("/generate").mock(return_value=httpx.Response(502))
        with pytest.raises(LLMProviderError):
            await llm_provider_client.generate_text("prompt")


class _FakeWebSocket:
    """Fakes the subset of the `websockets` client protocol `generate_description`
    relies on: `send(str)` and `async for message in ws`."""

    def __init__(self, events: list[str]):
        self._events = events
        self.sent: list[str] = []

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def __aiter__(self):
        for event in self._events:
            yield event


def _fake_connect(events: list[str]):
    @asynccontextmanager
    async def fake_connect(url, open_timeout=None):
        yield _FakeWebSocket(events)

    return fake_connect


async def test_generate_description_accumulates_chunks_from_ws(monkeypatch):
    events = [
        json.dumps({"type": "chunk", "text": "Ola "}),
        json.dumps({"type": "chunk", "text": "mundo"}),
        json.dumps({"type": "done", "text": "Ola mundo", "model": "m", "provider": "mock"}),
    ]
    monkeypatch.setattr(llm_provider_client.websockets, "connect", _fake_connect(events))

    result = await llm_provider_client.generate_description("prompt")
    assert result == "Ola mundo"


async def test_generate_description_raises_on_error_event(monkeypatch):
    events = [json.dumps({"type": "error", "message": "boom"})]
    monkeypatch.setattr(llm_provider_client.websockets, "connect", _fake_connect(events))

    with pytest.raises(LLMProviderError):
        await llm_provider_client.generate_description("prompt")
