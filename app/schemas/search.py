from pydantic import BaseModel

from app.schemas.product import ProductResponse


class SearchResultItem(BaseModel):
    product: ProductResponse
    score: float
    match_type: str


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]
    total: int
