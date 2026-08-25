from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models.product import DescriptionStatus, EmbeddingSyncStatus


class ProductCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200, examples=["Smartphone Galaxy X"])
    sku: str = Field(..., min_length=1, max_length=64, examples=["SKU-00123"])
    sale_type: str = Field(..., min_length=1, max_length=50, examples=["eletronico"])
    brand: str = Field(..., min_length=1, max_length=100, examples=["MarcaX"])
    category: str = Field(..., min_length=1, max_length=100, examples=["celulares"])
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional free-form metadata (color, voltage, fuel type, etc.)",
        examples=[{"cor": "preto", "memoria": "128GB"}],
    )


class ProductUpdate(BaseModel):
    """Partial update of product metadata. Only provided fields are changed."""

    name: str | None = Field(default=None, min_length=1, max_length=200)
    sale_type: str | None = Field(default=None, min_length=1, max_length=50)
    brand: str | None = Field(default=None, min_length=1, max_length=100)
    category: str | None = Field(default=None, min_length=1, max_length=100)
    attributes: dict[str, Any] | None = None


class ProductResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    sku: str
    sale_type: str
    brand: str
    category: str
    attributes: dict[str, Any]
    description: str | None
    suggested_description: str | None
    description_status: DescriptionStatus
    embedding_status: EmbeddingSyncStatus
    embedding_sync_error: str | None = None
    created_at: datetime
    updated_at: datetime


class ProductListResponse(BaseModel):
    items: list[ProductResponse]
    total: int
    skip: int
    limit: int


class ProductBatchRequest(BaseModel):
    ids: list[str] = Field(..., min_length=1, max_length=100)


class ProductBatchResponse(BaseModel):
    items: list[ProductResponse]


class DescriptionGenerateResponse(BaseModel):
    """A suggestion only: nothing is persisted until the user confirms it via PATCH."""

    product_id: str
    suggested_description: str
    description_status: DescriptionStatus


class DescriptionConfirmRequest(BaseModel):
    description: str = Field(..., min_length=1, max_length=5000)
