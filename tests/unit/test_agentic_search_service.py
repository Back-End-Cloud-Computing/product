from datetime import datetime, timezone

import pytest

from app.core.exceptions import AgenticSearchError
from app.services import agentic_search_service


def _make_doc(doc_id: str, name: str) -> dict:
    return {
        "_id": doc_id,
        "name": name,
        "sku": f"SKU-{doc_id}",
        "sale_type": "eletronico",
        "brand": "Marca",
        "category": "categoria",
        "attributes": {},
        "description": None,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }


class FakeLLMProviderClient:
    """Returns canned JSON based on which prompt is being asked (strategy,
    evaluation or justification), so the agentic loop can be driven
    deterministically without a real llm-provider call."""

    def __init__(self, strategy_responses=None, evaluation_responses=None, justification_response="[]"):
        self._strategies = list(strategy_responses or [])
        self._evaluations = list(evaluation_responses or [])
        self._justification = justification_response
        self.prompts: list[str] = []

    async def generate_text(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if "Estrategias possiveis" in prompt:
            return self._strategies.pop(0)
        if "Meta de documentos relevantes" in prompt:
            return self._evaluations.pop(0)
        return self._justification


def test_agregar_resultados_deduplicates_and_tracks_sources():
    state = agentic_search_service.AgenticSearchState(original_query="q")
    doc_a = _make_doc("a", "A")
    doc_b = _make_doc("b", "B")

    novos = agentic_search_service._agregar_resultados(state, [(doc_a, 1.0)], [(doc_a, 1.0), (doc_b, 1.0)])

    assert novos == 2
    assert state.found_documents["a"]["sources"] == {"lexical", "semantic"}
    assert state.found_documents["b"]["sources"] == {"semantic"}

    # Same doc seen again in a later iteration: score accumulates, but it is not "new".
    novos_again = agentic_search_service._agregar_resultados(state, [(doc_a, 1.0)], [])
    assert novos_again == 0
    assert state.found_documents["a"]["score"] > 0


async def test_loop_dynamically_switches_strategy_and_stops_when_satisfied(monkeypatch):
    doc_a = _make_doc("a", "Tenis A")
    doc_b = _make_doc("b", "Tenis B")

    lexical_calls: list[str] = []
    semantic_calls: list[str] = []

    async def fake_lexical(query, limit=10):
        lexical_calls.append(query)
        return [(doc_a, 2.0)]

    async def fake_semantic(query, limit=10):
        semantic_calls.append(query)
        return [(doc_b, 0.9)]

    monkeypatch.setattr(agentic_search_service.search_service, "search_lexical", fake_lexical)
    monkeypatch.setattr(agentic_search_service.search_service, "search_semantic", fake_semantic)

    provider = FakeLLMProviderClient(
        strategy_responses=['{"strategy": "lexical"}', '{"strategy": "semantic"}'],
        evaluation_responses=[
            '{"coverage": 0.5, "satisfied": false, "next_query": "tenis leve"}',
            '{"coverage": 1.0, "satisfied": true, "next_query": null}',
        ],
    )
    monkeypatch.setattr(agentic_search_service.llm_provider_client, "generate_text", provider.generate_text)

    events = [event async for event in agentic_search_service.buscar_agentica("tenis", limit=5, max_iterations=5)]

    # Iteration 1 only ran the lexical strategy against the original query;
    # iteration 2 only ran semantic, against the LLM-reformulated query.
    assert lexical_calls == ["tenis"]
    assert semantic_calls == ["tenis leve"]

    final = events[-1]
    assert final["type"] == "completed"
    assert final["iterations"] == 2
    assert final["coverage"] == 1.0

    results_by_id = {item["product"]["id"]: item for item in final["results"]}
    assert set(results_by_id) == {"a", "b"}
    assert results_by_id["a"]["source"] == "lexical"
    assert results_by_id["b"]["source"] == "semantic"


async def test_falls_back_to_hybrid_and_heuristic_when_llm_is_unavailable(monkeypatch):
    """The conftest-level fake `llm_provider_client.generate_text` returns fixed,
    non-JSON text (simulating an offline/mock llm-provider). Strategy/
    evaluation decisions must degrade gracefully instead of breaking the search."""
    doc_a = _make_doc("a", "Produto A")

    async def fake_lexical(query, limit=10):
        return [(doc_a, 1.0)]

    async def fake_semantic(query, limit=10):
        return []

    monkeypatch.setattr(agentic_search_service.search_service, "search_lexical", fake_lexical)
    monkeypatch.setattr(agentic_search_service.search_service, "search_semantic", fake_semantic)

    events = [
        event async for event in agentic_search_service.buscar_agentica("produto", limit=5, max_iterations=3)
    ]

    iteration_starts = [event for event in events if event["type"] == "iteration_start"]
    assert all(event["strategy"] == "hybrid" for event in iteration_starts)
    # Iteration 1 finds the one document (progress). Iteration 2 re-runs the same
    # (unchanged) query, finds nothing new, and the stagnation check stops the loop
    # there instead of spinning through every configured iteration for nothing.
    assert len(iteration_starts) == 2

    final = events[-1]
    assert final["type"] == "completed"
    assert final["results"][0]["source"] == "lexical"
    assert final["results"][0]["reason"]


async def test_max_iterations_is_capped_by_settings(monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "agentic_search_max_iterations", 1)

    call_count = 0

    async def growing_lexical(query, limit=10):
        nonlocal call_count
        call_count += 1
        return [(_make_doc(str(call_count), f"Produto {call_count}"), 1.0)]

    async def empty_semantic(query, limit=10):
        return []

    monkeypatch.setattr(agentic_search_service.search_service, "search_lexical", growing_lexical)
    monkeypatch.setattr(agentic_search_service.search_service, "search_semantic", empty_semantic)

    events = [event async for event in agentic_search_service.buscar_agentica("q", limit=5, max_iterations=5)]

    final = events[-1]
    assert final["type"] == "completed"
    assert final["iterations"] == 1


async def test_sync_wrapper_raises_agentic_search_error_on_failure(monkeypatch):
    async def failing_search(query, limit=10):
        raise RuntimeError("boom")

    monkeypatch.setattr(agentic_search_service.search_service, "search_lexical", failing_search)
    monkeypatch.setattr(agentic_search_service.search_service, "search_semantic", failing_search)

    with pytest.raises(AgenticSearchError):
        await agentic_search_service.buscar_agentica_sync("consulta", limit=5)
