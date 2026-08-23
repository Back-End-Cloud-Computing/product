from pydantic import BaseModel

from app.schemas.product import ProductResponse


class RecommendationItem(BaseModel):
    product: ProductResponse
    score: float
    reasons: list[str]


class RecommendationResponse(BaseModel):
    items: list[RecommendationItem]
    strategy: str
