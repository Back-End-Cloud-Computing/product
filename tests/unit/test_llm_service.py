from app.core import config
from app.services import llm_service


async def test_mock_provider_returns_non_empty_text():
    provider = llm_service.MockLLMProvider()
    text = await provider.generate_text("qualquer prompt")

    assert isinstance(text, str)
    assert len(text) > 0


def test_build_description_prompt_contains_metadata():
    prompt = llm_service.build_description_prompt(
        name="Produto A",
        brand="MarcaX",
        category="categoria",
        sale_type="eletronico",
        attributes={"cor": "azul"},
    )

    assert "Produto A" in prompt
    assert "MarcaX" in prompt
    assert "cor: azul" in prompt


def test_get_llm_provider_falls_back_to_mock_without_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    config.get_settings.cache_clear()

    try:
        provider = llm_service.get_llm_provider()
        assert isinstance(provider, llm_service.MockLLMProvider)
    finally:
        config.get_settings.cache_clear()
