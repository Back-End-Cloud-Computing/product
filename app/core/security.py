"""JWT authentication against the GANJJ `authorization` service.

Every request into this service must carry a valid access token, issued by
`authorization` and signed with RS256. Verification happens entirely locally
against the issuer's public key, fetched once at startup — there is no
per-request call to `authorization`. See
`authorization/docs/COMO-FUNCIONA.md` (in the authorization repo) for the
full rationale behind this design.
"""

import logging
from contextvars import ContextVar
from dataclasses import dataclass

import httpx
import jwt
import tenacity
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from fastapi import Header

from app.core.config import get_settings
from app.core.exceptions import AuthenticationError

logger = logging.getLogger(__name__)

ISSUER = "ganjj-authorization"
ACCESS_TOKEN_TYPE = "access"

_public_key: RSAPublicKey | None = None
_current_token: ContextVar[str | None] = ContextVar("current_token", default=None)


def _is_transient_fetch_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


@tenacity.retry(
    retry=tenacity.retry_if_exception(_is_transient_fetch_error),
    stop=tenacity.stop_after_delay(60),
    wait=tenacity.wait_fixed(2),
    reraise=True,
)
async def _fetch_public_key_pem() -> str:
    settings = get_settings()
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(f"{settings.auth_service_base_url}/auth/public-key")
        response.raise_for_status()
        return response.text


async def load_public_key() -> None:
    """Fetches and caches the authorization service's RSA public key.

    Called once from `lifespan`, before the app starts accepting requests.
    Retries transient failures (network/5xx) for up to 60s - `authorization`
    depends on Oracle and can take a while to come up. Without a public key
    no token can ever be validated, so a failure here is left to propagate
    and abort startup rather than let the service come up unable to do its
    one job.
    """
    global _public_key
    settings = get_settings()
    try:
        pem = await _fetch_public_key_pem()
    except (httpx.HTTPError, tenacity.RetryError) as exc:
        logger.error(
            "Could not fetch the authorization public key from %s after retrying for 60s. "
            "Is the authorization service up? (%s)",
            settings.auth_service_base_url,
            exc,
        )
        raise
    _public_key = serialization.load_pem_public_key(pem.encode())
    logger.info("Loaded authorization public key from %s", settings.auth_service_base_url)


@dataclass(frozen=True)
class CurrentUser:
    """The caller's identity, taken straight from the token's claims - no
    local user/client table is consulted or needed."""

    id: str
    email: str
    role: str


def auth_headers() -> dict[str, str]:
    """Authorization header to forward on outgoing calls to sibling GANJJ
    services, carrying the same access token this request came in with."""
    token = _current_token.get()
    return {"Authorization": f"Bearer {token}"} if token else {}


async def get_current_user(authorization: str | None = Header(default=None)) -> CurrentUser:
    """FastAPI dependency: validates the bearer token locally against the
    cached public key and returns the caller's identity. Wired once, for the
    whole router, in `main.py` - individual routes don't declare it."""
    if not authorization or not authorization.startswith("Bearer "):
        raise AuthenticationError("Missing or malformed Authorization header. Use: Bearer <token>.")

    token = authorization.removeprefix("Bearer ").strip()

    if _public_key is None:
        # Only reachable if load_public_key() never ran - normally startup
        # aborts before the app can serve any request.
        raise AuthenticationError("Authentication is not available: no public key loaded.")

    try:
        claims = jwt.decode(token, key=_public_key, algorithms=["RS256"], issuer=ISSUER)
    except jwt.PyJWTError as exc:
        raise AuthenticationError("Invalid or expired token.") from exc

    if claims.get("typ") != ACCESS_TOKEN_TYPE:
        raise AuthenticationError("This is not an access token.")

    try:
        user = CurrentUser(id=claims["sub"], email=claims["email"], role=claims["role"])
    except KeyError as exc:
        raise AuthenticationError("Token is missing required claims.") from exc

    _current_token.set(token)
    return user
