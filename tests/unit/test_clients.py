import json

import httpx
import pytest
import respx

from app.clients import embedding_reranking_client, llm_provider_client, vector_db_client
from app.core.config import get_settings
from app.core.exceptions import EmbeddingGenerationError, LLMProviderError
from app.core.security import _current_token


@pytest.fixture(autouse=True)
def patch_external_dependencies():
    """Overrides (shadows) the conftest-level autouse fixture of the same name:
    these tests exercise the real client implementations against a mocked HTTP
    transport (respx), so they must NOT have their target functions replaced
    with no-op fakes first."""
    yield


async def test_index_product_calls_embedding_reranking_with_expected_payload():
    settings = get_settings()
    with respx.mock(base_url=settings.embedding_reranking_base_url) as mock:
        route = mock.post("/index").mock(
            return_value=httpx.Response(
                200, json={"id": "1", "collection_name": "products", "status": "indexed", "model": "m"}
            )
        )
        await embedding_reranking_client.index_product("1", "some text", {"brand": "X"})

    assert route.called
    body = json.loads(route.calls.last.request.content)
    assert body == {"collection_name": "products", "id": "1", "text": "some text", "metadata": {"brand": "X"}}


async def test_index_product_raises_embedding_generation_error_on_failure():
    settings = get_settings()
    with respx.mock(base_url=settings.embedding_reranking_base_url) as mock:
        mock.post("/index").mock(return_value=httpx.Response(503))
        with pytest.raises(EmbeddingGenerationError):
            await embedding_reranking_client.index_product("1", "text", {})


async def test_embed_query_returns_first_embedding():
    settings = get_settings()
    with respx.mock(base_url=settings.embedding_reranking_base_url) as mock:
        route = mock.post("/embed").mock(
            return_value=httpx.Response(200, json={"embeddings": [[0.1, 0.2]], "model": "m", "count": 1})
        )
        result = await embedding_reranking_client.embed_query("tenis de corrida")

    assert result == [0.1, 0.2]
    body = json.loads(route.calls.last.request.content)
    assert body == {"texts": ["tenis de corrida"]}


async def test_embed_query_raises_embedding_generation_error_on_failure():
    settings = get_settings()
    with respx.mock(base_url=settings.embedding_reranking_base_url) as mock:
        mock.post("/embed").mock(return_value=httpx.Response(500))
        with pytest.raises(EmbeddingGenerationError):
            await embedding_reranking_client.embed_query("query")


async def test_rerank_returns_results_list():
    settings = get_settings()
    with respx.mock(base_url=settings.embedding_reranking_base_url) as mock:
        route = mock.post("/rerank").mock(
            return_value=httpx.Response(
                200,
                json={
                    "results": [{"passage": "doc", "score": 0.9, "index": 0}],
                    "model": "m",
                    "query": "query",
                },
            )
        )
        result = await embedding_reranking_client.rerank("query", ["doc"])

    assert result == [{"passage": "doc", "score": 0.9, "index": 0}]
    body = json.loads(route.calls.last.request.content)
    assert body == {"query": "query", "passages": ["doc"]}


async def test_rerank_raises_embedding_generation_error_on_failure():
    settings = get_settings()
    with respx.mock(base_url=settings.embedding_reranking_base_url) as mock:
        mock.post("/rerank").mock(return_value=httpx.Response(500))
        with pytest.raises(EmbeddingGenerationError):
            await embedding_reranking_client.rerank("query", ["doc"])


async def test_vector_db_search_returns_response_with_products_collection():
    settings = get_settings()
    with respx.mock(base_url=settings.vector_db_base_url) as mock:
        route = mock.post("/vector_db/search").mock(
            return_value=httpx.Response(
                200, json={"ids": ["1"], "distances": [0.1], "metadatas": [{}], "documents": ["doc"]}
            )
        )
        result = await vector_db_client.search([0.1, 0.2], n_results=5)

    assert result["ids"] == ["1"]
    body = json.loads(route.calls.last.request.content)
    assert body["collection_name"] == "products"
    assert body["embedding"] == [0.1, 0.2]


async def test_vector_db_search_raises_embedding_generation_error_on_failure():
    settings = get_settings()
    with respx.mock(base_url=settings.vector_db_base_url) as mock:
        mock.post("/vector_db/search").mock(return_value=httpx.Response(500))
        with pytest.raises(EmbeddingGenerationError):
            await vector_db_client.search([0.1, 0.2])


async def test_vector_db_delete_is_best_effort_and_does_not_raise():
    settings = get_settings()
    with respx.mock(base_url=settings.vector_db_base_url) as mock:
        route = mock.post("/vector_db/delete").mock(return_value=httpx.Response(500))
        await vector_db_client.delete_product("1")  # must not raise

    assert route.called
    body = json.loads(route.calls.last.request.content)
    assert body["collection_name"] == "products"


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


async def test_outgoing_calls_forward_the_incoming_bearer_token():
    """The token that authenticated this request into product-service must be
    forwarded as-is to the downstream services it calls on the caller's
    behalf - see app.core.security.auth_headers."""
    token = _current_token.set("the-caller-token")
    try:
        settings = get_settings()
        with respx.mock(base_url=settings.vector_db_base_url) as mock:
            route = mock.post("/vector_db/search").mock(
                return_value=httpx.Response(
                    200, json={"ids": [], "distances": [], "metadatas": [], "documents": []}
                )
            )
            await vector_db_client.search([0.1, 0.2])

        assert route.calls.last.request.headers["authorization"] == "Bearer the-caller-token"
    finally:
        _current_token.reset(token)
