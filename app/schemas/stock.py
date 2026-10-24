from pydantic import BaseModel, Field


class StockOperationRequest(BaseModel):
    quantity: int = Field(..., gt=0, examples=[2])
    reason: str | None = Field(default=None, max_length=100, examples=["order_checkout"])
    reference_id: str | None = Field(default=None, max_length=100, description="Caller's reference, e.g. a quote id.")
    compensates_operation_id: str | None = Field(
        default=None,
        max_length=100,
        description="On restore: the Idempotency-Key of the decrement being compensated.",
    )


class StockOperationResponse(BaseModel):
    operation_id: str
    product_id: str
    remaining_stock: int
