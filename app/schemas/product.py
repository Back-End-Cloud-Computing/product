from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    generate_description_with_ai: bool = Field(
        default=False,
        description=(
            "Se true, a descrição é gerada automaticamente pela LLM a partir dos "
            "demais campos e 'description' deve ser omitido. Se false, 'description' "
            "é obrigatório."
        ),
    )
    description: str | None = Field(
        default=None,
        min_length=1,
        max_length=5000,
        description="Obrigatório quando generate_description_with_ai=false; deve ser omitido quando true.",
        examples=["Smartphone com tela de 6.5 polegadas e 128GB de armazenamento."],
    )

    @model_validator(mode="after")
    def _validate_description_source(self) -> "ProductCreate":
        if self.generate_description_with_ai:
            if self.description is not None:
                raise ValueError("description must be omitted when generate_description_with_ai is true")
        elif not self.description:
            raise ValueError("description is required when generate_description_with_ai is false")
        return self


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
    summary: str | None
    error: str | None = None
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
