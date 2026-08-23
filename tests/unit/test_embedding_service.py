import pytest

from app.core.exceptions import EmbeddingGenerationError
from app.services import embedding_service


def test_build_semantic_text_combines_relevant_fields():
    text = embedding_service.build_semantic_text(
        name="Produto A",
        brand="MarcaX",
        category="categoria",
        description="uma boa descricao",
        attributes={"cor": "azul"},
    )

    assert "Produto A" in text
    assert "MarcaX" in text
    assert "cor: azul" in text


def test_build_semantic_text_skips_empty_parts():
    text = embedding_service.build_semantic_text(
        name="Produto A", brand="MarcaX", category="categoria", description="", attributes={}
    )

    assert text == "Produto A | MarcaX | categoria"


async def test_generate_embedding_wraps_backend_errors(monkeypatch):
    async def broken_model():
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(embedding_service, "get_embedding_model", broken_model)

    with pytest.raises(EmbeddingGenerationError):
        await embedding_service.generate_embedding("texto qualquer")
