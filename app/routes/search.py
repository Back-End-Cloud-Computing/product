import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from app.core.exceptions import EmbeddingGenerationError
from app.schemas.agentic_search import AgenticSearchRequest, AgenticSearchResponse
from app.schemas.search import SearchResponse, SearchResultItem
from app.services import agentic_search_service, product_service, search_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


@router.get("/lexical", response_model=SearchResponse)
async def search_lexical(q: str = Query(..., min_length=1), limit: int = Query(10, ge=1, le=50)) -> SearchResponse:
    results = await search_service.search_lexical(q, limit=limit)
    items = [
        SearchResultItem(product=product_service.document_to_response(doc), score=score, match_type="lexical")
        for doc, score in results
    ]
    return SearchResponse(query=q, results=items, total=len(items))


@router.get("/semantic", response_model=SearchResponse)
async def search_semantic(q: str = Query(..., min_length=1), limit: int = Query(10, ge=1, le=50)) -> SearchResponse:
   
    try:
        results = await search_service.search_semantic(q, limit=limit)
        match_type = "semantic"
    except EmbeddingGenerationError as exc:
        logger.warning("Semantic search unavailable (%s); falling back to lexical for query '%s'", exc, q)
        results = await search_service.search_lexical(q, limit=limit)
        match_type = "lexical_fallback"

    items = [
        SearchResultItem(product=product_service.document_to_response(doc), score=score, match_type=match_type)
        for doc, score in results
    ]
    return SearchResponse(query=q, results=items, total=len(items))


@router.post("/agentic_search", response_model=AgenticSearchResponse)
async def agentic_search(payload: AgenticSearchRequest) -> AgenticSearchResponse:
    """Busca agentica: substitui a antiga busca hibrida (RRF fixo). A cada
    iteracao a LLM decide entre busca full-text, semantica ou ambas, agrega/
    deduplica os resultados e avalia se a cobertura ja e suficiente,
    reformulando a consulta quando necessario. Roda a mesma logica do
    WebSocket abaixo, sem o streaming de progresso por iteracao."""
    evento = await agentic_search_service.buscar_agentica_sync(
        payload.query, limit=payload.limit, max_iterations=payload.max_iterations
    )
    return AgenticSearchResponse(**evento)


@router.websocket("/agentic_search/ws")
async def agentic_search_ws(websocket: WebSocket) -> None:
    """Mesma busca agentica do POST, transmitindo o progresso de cada iteracao
    (estrategia escolhida, documentos novos, cobertura estimada) antes de
    enviar o evento final com os documentos considerados relevantes."""
    await websocket.accept()
    try:
        payload = AgenticSearchRequest.model_validate_json(await websocket.receive_text())
        async for evento in agentic_search_service.buscar_agentica(
            payload.query, limit=payload.limit, max_iterations=payload.max_iterations
        ):
            await websocket.send_json(evento)
            if evento["type"] in ("completed", "error"):
                break
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001 - a malformed payload must not crash the connection silently
        logger.error("Agentic search websocket failed: %s", exc)
        try:
            await websocket.send_json({"type": "error", "message": "Erro ao realizar busca agentica"})
        except Exception:  # noqa: BLE001 - connection may already be gone
            pass
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass
