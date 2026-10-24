"""Service-to-service routes. Authenticated by the shared `x-internal-secret`
header instead of a user JWT, mounted outside the public router in
`main.py`, and left out of the Ingress: only callers inside the cluster
(order's checkout saga) reach them."""

from fastapi import APIRouter, Depends, Header

from app.core.security import require_internal_secret
from app.schemas.stock import StockOperationRequest, StockOperationResponse
from app.services import stock_service

router = APIRouter(
    prefix="/internal/products",
    tags=["internal"],
    dependencies=[Depends(require_internal_secret)],
)


@router.post("/{product_id}/stock/decrement", response_model=StockOperationResponse)
async def decrement_stock(
    product_id: str,
    payload: StockOperationRequest,
    idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1, max_length=100),
) -> StockOperationResponse:
    return await stock_service.apply_stock_operation(product_id, "decrement", payload.quantity, idempotency_key)


@router.post("/{product_id}/stock/restore", response_model=StockOperationResponse)
async def restore_stock(
    product_id: str,
    payload: StockOperationRequest,
    idempotency_key: str = Header(..., alias="Idempotency-Key", min_length=1, max_length=100),
) -> StockOperationResponse:
    return await stock_service.apply_stock_operation(product_id, "restore", payload.quantity, idempotency_key)
