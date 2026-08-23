from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DescriptionStatus(str, Enum):
    PENDING = "pending"
    SUGGESTED = "suggested"
    APPROVED = "approved"


class EmbeddingSyncStatus(str, Enum):
    PENDING = "pending"
    SYNCED = "synced"
    FAILED = "failed"


class ProductDocument(BaseModel):
    """Persistence representation of a product as stored in MongoDB.

    This is intentionally distinct from the API schemas in `app.schemas.product`:
    it carries internal lifecycle fields (description/embedding sync status) that
    are never part of the request contracts.
    """

    name: str
    sku: str
    sale_type: str
    brand: str
    category: str
    attributes: dict[str, Any] = Field(default_factory=dict)

    description: str | None = None
    suggested_description: str | None = None
    description_status: DescriptionStatus = DescriptionStatus.PENDING

    embedding_status: EmbeddingSyncStatus = EmbeddingSyncStatus.PENDING
    embedding_sync_error: str | None = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_mongo(self) -> dict[str, Any]:
        return self.model_dump()
