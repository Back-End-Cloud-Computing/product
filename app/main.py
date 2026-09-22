from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.exceptions import (
    AgenticSearchError,
    AuthenticationError,
    DuplicateSkuError,
    EmbeddingGenerationError,
    LLMProviderError,
    ProductNotFoundError,
)
from app.core.logging import configure_logging
from app.core.security import get_current_user, load_public_key
from app.database import mongodb
from app.routes import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    await mongodb.connect_to_mongo()
    await load_public_key()
    yield
    await mongodb.close_mongo_connection()


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        description=(
            "Microsservico de Produto: CRUD, geracao de descricao comercial via llm-provider, "
            "busca lexical/semantica/agentica (semantica e agentica via embedding-reranking)."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    application.include_router(api_router, dependencies=[Depends(get_current_user)])

    @application.exception_handler(AuthenticationError)
    async def authentication_error_handler(request: Request, exc: AuthenticationError) -> JSONResponse:
        return JSONResponse(status_code=401, content={"detail": str(exc), "error_type": "authentication_error"})

    @application.exception_handler(ProductNotFoundError)
    async def product_not_found_handler(request: Request, exc: ProductNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc), "error_type": "product_not_found"})

    @application.exception_handler(DuplicateSkuError)
    async def duplicate_sku_handler(request: Request, exc: DuplicateSkuError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc), "error_type": "duplicate_sku"})

    @application.exception_handler(LLMProviderError)
    async def llm_provider_error_handler(request: Request, exc: LLMProviderError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc), "error_type": "llm_provider_error"})

    @application.exception_handler(EmbeddingGenerationError)
    async def embedding_error_handler(request: Request, exc: EmbeddingGenerationError) -> JSONResponse:
        return JSONResponse(
            status_code=503, content={"detail": str(exc), "error_type": "embedding_generation_error"}
        )

    @application.exception_handler(AgenticSearchError)
    async def agentic_search_error_handler(request: Request, exc: AgenticSearchError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc), "error_type": "agentic_search_error"})

    @application.get("/health", tags=["health"])
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    return application


app = create_app()
