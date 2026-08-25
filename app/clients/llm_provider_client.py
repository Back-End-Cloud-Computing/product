import json
import logging
from typing import Any

import httpx
import websockets

from app.core.config import get_settings
from app.core.exceptions import LLMProviderError

logger = logging.getLogger(__name__)


async def generate_text(prompt: str) -> str:
    """Plain REST call to llm-provider. Used for every LLM call that is not
    description generation (agentic search's strategy/evaluation/justification
    prompts), which stays on HTTPS rather than the WS transport."""
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


async def generate_description(prompt: str) -> str:
    """Description generation is the one flow required to use WebSocket
    streaming between product-service and llm-provider. Chunks are collected
    into the full suggestion; the external product-service endpoint stays
    plain REST, so streaming here is internal only."""
    settings = get_settings()
    url = f"{settings.llm_provider_ws_url}/generate/ws"
    try:
        async with websockets.connect(url, open_timeout=settings.llm_provider_timeout_seconds) as ws:
            await ws.send(json.dumps({"prompt": prompt}))
            chunks: list[str] = []
            async for raw_message in ws:
                event = json.loads(raw_message)
                if event["type"] == "chunk":
                    chunks.append(event["text"])
                elif event["type"] == "done":
                    return event.get("text") or "".join(chunks)
                elif event["type"] == "error":
                    raise LLMProviderError(f"llm-provider streaming failed: {event.get('message')}")
    except (OSError, TimeoutError) as exc:
        logger.error("llm-provider WS connection failed: %s", exc)
        raise LLMProviderError(f"llm-provider WS connection failed: {exc}") from exc
    raise LLMProviderError("llm-provider WS closed without a terminal event")
