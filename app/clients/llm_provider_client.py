import logging
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.exceptions import LLMProviderError

logger = logging.getLogger(__name__)


async def generate_text(prompt: str) -> str:
    """Plain REST call to llm-provider. Used for every LLM call this service
    makes: description generation, summary generation, and agentic search's
    strategy/evaluation/justification prompts."""
    settings = get_settings()
    payload: dict[str, Any] = {"prompt": prompt}
    try:
        async with httpx.AsyncClient(timeout=settings.llm_provider_timeout_seconds) as client:
            response = await client.post(f"{settings.llm_provider_base_url}/generate", json=payload)
            response.raise_for_status()
            return response.json()["text"]
    except (httpx.HTTPError, KeyError) as exc:
        logger.error("llm-provider /generate failed: %s", exc)
        raise LLMProviderError(f"llm-provider request failed: {exc}") from exc
