from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.exceptions import (
    DescriptionNotApprovedError,
    DuplicateSkuError,
    EmbeddingGenerationError,
    LLMProviderError,
    ProductNotFoundError,
    VectorStoreError,
)
from app.core.logging import configure_logging
from app.database import chromadb_client, mongodb
from app.routes import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    await mongodb.connect_to_mongo()
    chromadb_client.connect_to_chromadb()
    yield
    await mongodb.close_mongo_connection()


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        description=(
            "Microsservico de Produto: cadastro, geracao de descricao comercial via LLM, "
            "embeddings, busca lexical/semantica/hibrida e recomendacao."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    application.include_router(api_router)

    @application.exception_handler(ProductNotFoundError)
    async def product_not_found_handler(request: Request, exc: ProductNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc), "error_type": "product_not_found"})

    @application.exception_handler(DuplicateSkuError)
    async def duplicate_sku_handler(request: Request, exc: DuplicateSkuError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc), "error_type": "duplicate_sku"})

    @application.exception_handler(DescriptionNotApprovedError)
    async def description_not_approved_handler(request: Request, exc: DescriptionNotApprovedError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc), "error_type": "description_not_approved"})

    @application.exception_handler(LLMProviderError)
    async def llm_provider_error_handler(request: Request, exc: LLMProviderError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc), "error_type": "llm_provider_error"})

    @application.exception_handler(EmbeddingGenerationError)
    async def embedding_error_handler(request: Request, exc: EmbeddingGenerationError) -> JSONResponse:
        return JSONResponse(
            status_code=503, content={"detail": str(exc), "error_type": "embedding_generation_error"}
        )

    @application.exception_handler(VectorStoreError)
    async def vector_store_error_handler(request: Request, exc: VectorStoreError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc), "error_type": "vector_store_error"})

    @application.get("/health", tags=["health"])
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    return application


app = create_app()
