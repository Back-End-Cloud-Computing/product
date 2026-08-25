from fastapi import APIRouter, Query, status

from app.schemas.product import (
    DescriptionConfirmRequest,
    DescriptionGenerateResponse,
    ProductBatchRequest,
    ProductBatchResponse,
    ProductCreate,
    ProductListResponse,
    ProductResponse,
    ProductUpdate,
)
from app.services import product_service

router = APIRouter(prefix="/products", tags=["products"])


@router.post("", response_model=ProductResponse, status_code=status.HTTP_201_CREATED)
async def create_product(payload: ProductCreate) -> ProductResponse:
    return await product_service.create_product(payload)


@router.post("/batch", response_model=ProductBatchResponse)
async def get_products_batch(payload: ProductBatchRequest) -> ProductBatchResponse:
    """Hydrates a batch of product ids. Meant for callers (e.g. a future
    frontend/gateway) that got thin `product_id` + `score` results from
    vector-db's recommendation endpoints and need the full product data."""
    items = await product_service.get_products_batch(payload.ids)
    return ProductBatchResponse(items=items)


@router.get("", response_model=ProductListResponse)
async def list_products(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    category: str | None = None,
    brand: str | None = None,
) -> ProductListResponse:
    items, total = await product_service.list_products(skip=skip, limit=limit, category=category, brand=brand)
    return ProductListResponse(items=items, total=total, skip=skip, limit=limit)


@router.get("/{product_id}", response_model=ProductResponse)
async def get_product(product_id: str) -> ProductResponse:
    return await product_service.get_product(product_id)


@router.patch("/{product_id}", response_model=ProductResponse)
async def update_product(product_id: str, payload: ProductUpdate) -> ProductResponse:
    return await product_service.update_product(product_id, payload)


@router.delete("/{product_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_product(product_id: str) -> None:
    await product_service.delete_product(product_id)


@router.post("/{product_id}/description/generate", response_model=DescriptionGenerateResponse)
async def generate_description(product_id: str) -> DescriptionGenerateResponse:
    """Ask the LLM for a description suggestion. This is a preview only —
    nothing is persisted as the final description until the client PATCHes
    `/products/{product_id}/description` with the (possibly edited) text."""
    product = await product_service.generate_description_suggestion(product_id)
    return DescriptionGenerateResponse(
        product_id=product.id,
        suggested_description=product.suggested_description or "",
        description_status=product.description_status,
    )


@router.patch("/{product_id}/description", response_model=ProductResponse)
async def confirm_description(product_id: str, payload: DescriptionConfirmRequest) -> ProductResponse:
    """Persist the final, user-approved description and trigger embedding sync."""
    return await product_service.confirm_description(product_id, payload.description)


@router.post("/{product_id}/embedding/sync", response_model=ProductResponse)
async def retry_embedding_sync(product_id: str) -> ProductResponse:
    """Manually retry a failed MongoDB -> ChromaDB embedding sync."""
    return await product_service.retry_embedding_sync(product_id)
