import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from app.core.config import get_settings
from app.core.exceptions import AgenticSearchError, LLMProviderError
from app.services import llm_service, product_service, search_service

logger = logging.getLogger(__name__)

# Reciprocal Rank Fusion constant: flattens the influence of rank position so a
# doc surfaced by only one strategy isn't swamped by ones found by both, and
# scores stay comparable across iterations.
RRF_K = 60

_VALID_STRATEGIES = {"lexical", "semantic", "hybrid"}


@dataclass
class AgenticSearchState:
    """Execution state threaded through the agentic loop.

    `found_documents` doubles as the dedup/aggregation structure: keyed by
    product id, it accumulates a fused score and the set of strategies that
    surfaced that document across every iteration so far.
    """

    original_query: str
    query_history: list[str] = field(default_factory=list)
    found_documents: dict[str, dict[str, Any]] = field(default_factory=dict)
    iteration: int = 0
    coverage: float = 0.0


def _parse_json_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_json_array(text: str) -> list[dict[str, Any]] | None:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, list) else None


def _build_strategy_prompt(state: AgenticSearchState, current_query: str) -> str:
    history_text = " -> ".join(state.query_history) or current_query
    return (
        "Voce e um agente de busca de produtos em um e-commerce. Decida a melhor "
        "estrategia de busca para a consulta atual.\n\n"
        f"Consulta original do usuario: {state.original_query}\n"
        f"Consulta atual (iteracao {state.iteration}): {current_query}\n"
        f"Historico de consultas ja tentadas: {history_text}\n"
        f"Documentos ja encontrados ate agora: {len(state.found_documents)}\n\n"
        "Estrategias possiveis:\n"
        '- "lexical": correspondencia exata de palavras-chave (bom para SKU, marca, termos exatos)\n'
        '- "semantic": busca por significado/intencao (bom para descricoes vagas ou sinonimos)\n'
        '- "hybrid": combina as duas, util quando ha duvida sobre a natureza da consulta\n\n'
        "Responda APENAS com um JSON no formato exato, sem nenhum texto adicional: "
        '{"strategy": "lexical" | "semantic" | "hybrid"}'
    )


def _build_evaluation_prompt(state: AgenticSearchState, limit: int) -> str:
    amostra = list(state.found_documents.values())[:8]
    documentos_texto = "\n".join(
        f"- {item['doc'].get('name', 'sem nome')}: "
        f"{item['doc'].get('description') or item['doc'].get('category', '')}"
        for item in amostra
    ) or "nenhum documento encontrado ate agora"

    return (
        "Voce e um agente de busca avaliando se os documentos encontrados respondem "
        "adequadamente a pergunta do usuario.\n\n"
        f"Pergunta original: {state.original_query}\n"
        f"Documentos encontrados ({len(state.found_documents)} no total, amostra abaixo):\n"
        f"{documentos_texto}\n\n"
        f"Meta de documentos relevantes: {limit}\n\n"
        "Avalie a cobertura atual e decida se a busca deve continuar. Caso deva continuar, "
        "sugira uma consulta reformulada (novos termos, sinonimos ou mais especifica) para "
        "a proxima iteracao. Responda APENAS com um JSON no formato exato, sem nenhum texto "
        "adicional: "
        '{"coverage": <numero de 0.0 a 1.0>, "satisfied": true | false, '
        '"next_query": "<consulta reformulada ou null>"}'
    )


def _build_justification_prompt(query: str, documentos: list[dict[str, Any]]) -> str:
    listagem = "\n".join(
        f"- id={item['id']}: {item['doc'].get('name', '')} | {item['doc'].get('category', '')} | "
        f"{(item['doc'].get('description') or '')[:200]}"
        for item in documentos
    )
    return (
        "Voce e um agente de busca de produtos. Para cada documento abaixo, escreva uma "
        "unica frase curta (em portugues) explicando por que ele e relevante para a pergunta "
        "do usuario.\n\n"
        f"Pergunta do usuario: {query}\n\n"
        f"Documentos:\n{listagem}\n\n"
        "Responda APENAS com um JSON (lista) no formato exato, sem nenhum texto adicional: "
        '[{"id": "<id>", "reason": "<frase curta>"}]'
    )


def _default_reason(sources: set[str], query: str) -> str:
    if sources == {"lexical", "semantic"}:
        return f"Encontrado tanto por correspondencia textual quanto por similaridade semantica com '{query}'."
    if "lexical" in sources:
        return f"Encontrado por correspondencia textual direta com os termos de '{query}'."
    return f"Encontrado por similaridade semantica com a intencao de '{query}'."


def _final_source(sources: set[str]) -> str:
    if sources == {"lexical", "semantic"}:
        return "both"
    return next(iter(sources))


async def _decidir_estrategia(provider: llm_service.LLMProvider, state: AgenticSearchState, current_query: str) -> str:
    """Asks the LLM which strategy to use this iteration. Any failure to reach the
    provider, or a response that isn't parseable JSON with a known strategy,
    degrades to "hybrid" so the loop keeps making forward progress offline."""
    prompt = _build_strategy_prompt(state, current_query)
    try:
        raw = await provider.generate_text(prompt)
    except LLMProviderError:
        return "hybrid"

    parsed = _parse_json_object(raw)
    strategy = parsed.get("strategy") if parsed else None
    return strategy if strategy in _VALID_STRATEGIES else "hybrid"


async def _avaliar_cobertura(
    provider: llm_service.LLMProvider, state: AgenticSearchState, limit: int
) -> dict[str, Any]:
    """Asks the LLM whether the documents found so far satisfy the user's intent.
    Falls back to a simple document-count heuristic when the provider is
    unreachable or answers with something unparseable."""
    heuristica = {
        "coverage": min(1.0, len(state.found_documents) / limit) if limit else 0.0,
        "satisfied": len(state.found_documents) >= limit,
        "next_query": None,
    }

    prompt = _build_evaluation_prompt(state, limit)
    try:
        raw = await provider.generate_text(prompt)
    except LLMProviderError:
        return heuristica

    parsed = _parse_json_object(raw)
    if not parsed or "satisfied" not in parsed:
        return heuristica

    try:
        coverage = float(parsed.get("coverage", heuristica["coverage"]))
    except (TypeError, ValueError):
        coverage = heuristica["coverage"]

    return {
        "coverage": max(0.0, min(1.0, coverage)),
        "satisfied": bool(parsed.get("satisfied")),
        "next_query": parsed.get("next_query") or None,
    }


async def _executar_estrategia(
    strategy: str, query: str, limit: int
) -> tuple[list[tuple[dict[str, Any], float]], list[tuple[dict[str, Any], float]]]:
    lexical_results: list[tuple[dict[str, Any], float]] = []
    semantic_results: list[tuple[dict[str, Any], float]] = []
    if strategy in ("lexical", "hybrid"):
        lexical_results = await search_service.search_lexical(query, limit=limit * 2)
    if strategy in ("semantic", "hybrid"):
        semantic_results = await search_service.search_semantic(query, limit=limit * 2)
    return lexical_results, semantic_results


def _agregar_resultados(
    state: AgenticSearchState,
    lexical_results: list[tuple[dict[str, Any], float]],
    semantic_results: list[tuple[dict[str, Any], float]],
) -> int:
    """Fuses new results into state.found_documents by rank (RRF), deduplicating
    by product id, and returns how many *new* documents this iteration added
    (used to detect stagnation and stop early)."""
    novos = 0

    def _somar(results: list[tuple[dict[str, Any], float]], origem: str) -> None:
        nonlocal novos
        for rank, (doc, _score) in enumerate(results):
            doc_id = str(doc["_id"])
            if doc_id not in state.found_documents:
                novos += 1
                state.found_documents[doc_id] = {"doc": doc, "score": 0.0, "sources": set()}
            state.found_documents[doc_id]["score"] += 1.0 / (RRF_K + rank + 1)
            state.found_documents[doc_id]["sources"].add(origem)

    _somar(lexical_results, "lexical")
    _somar(semantic_results, "semantic")
    return novos


async def _gerar_justificativas(
    provider: llm_service.LLMProvider, query: str, documentos: list[dict[str, Any]]
) -> dict[str, str]:
    if not documentos:
        return {}

    prompt = _build_justification_prompt(query, documentos)
    parsed: list[dict[str, Any]] | None = None
    try:
        raw = await provider.generate_text(prompt)
        parsed = _parse_json_array(raw)
    except LLMProviderError:
        parsed = None

    justificativas: dict[str, str] = {}
    if parsed:
        for item in parsed:
            doc_id = item.get("id")
            motivo = item.get("reason")
            if doc_id and motivo:
                justificativas[str(doc_id)] = str(motivo)

    for item in documentos:
        justificativas.setdefault(item["id"], _default_reason(item["sources"], query))
    return justificativas


async def buscar_agentica(
    query: str,
    limit: int = 10,
    max_iterations: int | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Core of the agentic search: each iteration the LLM decides between lexical,
    semantic or both, results are fused/deduplicated into the running state, and
    the LLM evaluates whether coverage is sufficient or the query should be
    reformulated for another pass. Shared by both the POST and WebSocket
    endpoints so the two transports can never diverge in behavior.

    Yields progress events (`iteration_start`, `iteration_result`) followed by
    exactly one terminal event (`completed` or `error`).
    """
    settings = get_settings()
    max_iter = (
        min(max_iterations, settings.agentic_search_max_iterations)
        if max_iterations
        else settings.agentic_search_max_iterations
    )

    state = AgenticSearchState(original_query=query)
    current_query = query

    try:
        provider = llm_service.get_llm_provider()

        while state.iteration < max_iter:
            state.iteration += 1
            state.query_history.append(current_query)

            strategy = await _decidir_estrategia(provider, state, current_query)
            yield {
                "type": "iteration_start",
                "iteration": state.iteration,
                "strategy": strategy,
                "query": current_query,
            }

            lexical_results, semantic_results = await _executar_estrategia(strategy, current_query, limit)
            novos = _agregar_resultados(state, lexical_results, semantic_results)

            avaliacao = await _avaliar_cobertura(provider, state, limit)
            state.coverage = avaliacao["coverage"]

            yield {
                "type": "iteration_result",
                "iteration": state.iteration,
                "strategy": strategy,
                "new_documents": novos,
                "total_documents": len(state.found_documents),
                "coverage": state.coverage,
            }

            if avaliacao["satisfied"] or novos == 0:
                break
            current_query = avaliacao["next_query"] or current_query

        ranked = sorted(state.found_documents.items(), key=lambda kv: kv[1]["score"], reverse=True)[:limit]
        documentos_selecionados = [
            {"id": doc_id, "doc": info["doc"], "score": info["score"], "sources": info["sources"]}
            for doc_id, info in ranked
        ]
        justificativas = await _gerar_justificativas(provider, state.original_query, documentos_selecionados)

        resultados = [
            {
                "product": product_service.document_to_response(item["doc"]).model_dump(mode="json"),
                "source": _final_source(item["sources"]),
                "score": round(item["score"], 4),
                "reason": justificativas.get(item["id"], _default_reason(item["sources"], state.original_query)),
            }
            for item in documentos_selecionados
        ]

        yield {
            "type": "completed",
            "query": state.original_query,
            "results": resultados,
            "total": len(resultados),
            "iterations": state.iteration,
            "coverage": state.coverage,
        }
    except Exception as exc:  
        logger.error("Agentic search failed for query '%s': %s", query, exc)
        yield {"type": "error", "message": "Erro ao realizar busca agentica"}


async def buscar_agentica_sync(
    query: str,
    limit: int = 10,
    max_iterations: int | None = None,
) -> dict[str, Any]:
    """Drains the event generator and returns only the terminal event, for
    callers (the POST endpoint) that don't need iteration-by-iteration
    progress."""
    async for evento in buscar_agentica(query, limit=limit, max_iterations=max_iterations):
        if evento["type"] == "completed":
            evento.pop("type", None)
            return evento
        if evento["type"] == "error":
            raise AgenticSearchError(evento["message"])
    raise AgenticSearchError("Erro ao realizar busca agentica")
