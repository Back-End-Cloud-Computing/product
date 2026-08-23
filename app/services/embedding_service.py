import asyncio
import logging
from typing import Any

from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.exceptions import EmbeddingGenerationError

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None
_model_lock = asyncio.Lock()


async def get_embedding_model() -> SentenceTransformer:
    """Lazily load the embedding model once per process (CPU-bound, so it runs
    in a worker thread to avoid blocking the event loop)."""
    global _model
    if _model is None:
        async with _model_lock:
            if _model is None:
                settings = get_settings()
                logger.info("Loading embedding model '%s'", settings.embedding_model_name)
                _model = await asyncio.to_thread(
                    SentenceTransformer,
                    settings.embedding_model_name,
                    device=settings.embedding_device,
                )
    return _model


def build_semantic_text(
    name: str,
    brand: str,
    category: str,
    description: str,
    attributes: dict[str, Any],
) -> str:
    """Combine the semantically relevant product fields into a single text blob
    used both as the embedding input and as the ChromaDB stored document."""
    attributes_text = " ".join(f"{key}: {value}" for key, value in attributes.items())
    parts = [name, brand, category, description, attributes_text]
    return " | ".join(part for part in parts if part)


async def generate_embedding(text: str) -> list[float]:
    try:
        model = await get_embedding_model()
        embedding = await asyncio.to_thread(model.encode, text, normalize_embeddings=True)
        return embedding.tolist()
    except Exception as exc:  # noqa: BLE001 - normalize any backend failure
        raise EmbeddingGenerationError(f"Failed to generate embedding: {exc}") from exc
