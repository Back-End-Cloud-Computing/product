import logging
from abc import ABC, abstractmethod
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.exceptions import LLMProviderError

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstraction over the external LLM used to draft product descriptions.

    Keeping this behind an interface means the concrete provider (and its cost,
    availability, and latency characteristics) can be swapped via configuration
    without touching the callers, and the service can run/test fully offline
    via `MockLLMProvider`.
    """

    @abstractmethod
    async def generate_text(self, prompt: str) -> str:
        """Generate text from a prompt. Must raise LLMProviderError on failure."""


class OpenRouterLLMProvider(LLMProvider):
    """Calls the OpenRouter chat-completions API (OpenAI-compatible)."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.openrouter_api_key:
            raise LLMProviderError("OPENROUTER_API_KEY is not configured")
        self._model = settings.llm_model
        self._base_url = settings.openrouter_base_url
        self._api_key = settings.openrouter_api_key
        self._timeout = settings.llm_timeout_seconds

    async def generate_text(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 400,
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
        except (httpx.HTTPError, KeyError, IndexError, TypeError) as exc:
            logger.error("OpenRouter request failed: %s", exc)
            raise LLMProviderError(f"OpenRouter request failed: {exc}") from exc


class MockLLMProvider(LLMProvider):
    """Deterministic, offline fallback used when no LLM provider is configured,
    reachable, or when running tests. Never raises."""

    async def generate_text(self, prompt: str) -> str:
        return (
            "Descrição gerada automaticamente em modo offline (nenhum provedor de LLM "
            "configurado ou disponível no momento). Edite este texto livremente antes "
            "de aprovar o produto."
        )


def get_llm_provider() -> LLMProvider:
    settings = get_settings()
    if settings.llm_provider == "openrouter":
        try:
            return OpenRouterLLMProvider()
        except LLMProviderError:
            logger.warning("OpenRouter is not configured; falling back to MockLLMProvider")
            return MockLLMProvider()
    return MockLLMProvider()


def build_description_prompt(
    name: str,
    brand: str,
    category: str,
    sale_type: str,
    attributes: dict[str, Any],
) -> str:
    attributes_text = ", ".join(f"{key}: {value}" for key, value in attributes.items())
    if not attributes_text:
        attributes_text = "nenhum atributo adicional informado"

    return (
        "Você é um redator de e-commerce. Escreva uma descrição comercial persuasiva e "
        "objetiva, em português, para o produto abaixo. Use no máximo 3 parágrafos curtos "
        "e não invente características que não foram informadas.\n\n"
        f"Nome: {name}\n"
        f"Marca: {brand}\n"
        f"Categoria: {category}\n"
        f"Tipo de venda: {sale_type}\n"
        f"Atributos: {attributes_text}\n"
    )
