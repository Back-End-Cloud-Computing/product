from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class ProductDocument(BaseModel):
    """Persistence representation of a product as stored in MongoDB.

    This is intentionally distinct from the API schemas in `app.schemas.product`:
    it carries internal fields (`summary`, `error`) that are never part of the
    request contracts.
    """

    name: str
    sku: str
    sale_type: str
    brand: str
    category: str
    attributes: dict[str, Any] = Field(default_factory=dict)

    description: str | None = None
    summary: str | None = None

    error: str | None = None
    """Set to the exception type name of the first failure in the post-creation
    pipeline (AI description generation, summary generation, or vector
    indexing). `None` means every step succeeded. There is no retry: creation
    is best-effort and this field is purely informational."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_mongo(self) -> dict[str, Any]:
        return self.model_dump()
