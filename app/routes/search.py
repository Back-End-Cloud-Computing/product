from fastapi import APIRouter, Query

from app.schemas.search import SearchResponse, SearchResultItem
from app.services import product_service, search_service

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
    results = await search_service.search_semantic(q, limit=limit)
    items = [
        SearchResultItem(product=product_service.document_to_response(doc), score=score, match_type="semantic")
        for doc, score in results
    ]
    return SearchResponse(query=q, results=items, total=len(items))


@router.get("/hybrid", response_model=SearchResponse)
async def search_hybrid(q: str = Query(..., min_length=1), limit: int = Query(10, ge=1, le=50)) -> SearchResponse:
    results = await search_service.search_hybrid(q, limit=limit)
    items = [
        SearchResultItem(product=product_service.document_to_response(doc), score=score, match_type="hybrid")
        for doc, score in results
    ]
    return SearchResponse(query=q, results=items, total=len(items))
