from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.product import ProductResponse

SearchStrategy = Literal["lexical", "semantic", "hybrid"]

SearchSource = Literal["lexical", "semantic", "both"]


class AgenticSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, examples=["tenis de corrida leve para maratona"])
    limit: int = Field(default=10, ge=1, le=50)
    max_iterations: int | None = Field(default=None, ge=1, le=10)


class AgenticSearchResultItem(BaseModel):
    product: ProductResponse
    source: SearchSource
    score: float
    reason: str


class AgenticSearchResponse(BaseModel):
    query: str
    results: list[AgenticSearchResultItem]
    total: int
    iterations: int
    coverage: float
