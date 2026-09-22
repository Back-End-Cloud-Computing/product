"""End-to-end check that routes are actually protected, exercised against the
real `get_current_user` dependency - unlike the rest of the integration
suite, which uses the `api_client` fixture's dependency override."""

import time

import jwt
import pytest_asyncio
from cryptography.hazmat.primitives.asymmetric import rsa
from httpx import ASGITransport, AsyncClient

from app.core import security


@pytest_asyncio.fixture
async def real_auth_client(mongo_database, monkeypatch):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    monkeypatch.setattr(security, "_public_key", private_key.public_key())

    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, private_key


def _valid_token(private_key) -> str:
    now = int(time.time())
    claims = {
        "iss": security.ISSUER,
        "sub": "11111111-1111-1111-1111-111111111111",
        "email": "teste@ganjj.com",
        "role": "CLIENTE",
        "typ": "access",
        "iat": now,
        "exp": now + 3600,
    }
    return jwt.encode(claims, private_key, algorithm="RS256")


async def test_protected_route_without_token_returns_401(real_auth_client):
    client, _ = real_auth_client
    response = await client.get("/products")
    assert response.status_code == 401
    assert response.json()["error_type"] == "authentication_error"


async def test_protected_route_with_malformed_header_returns_401(real_auth_client):
    client, _ = real_auth_client
    response = await client.get("/products", headers={"Authorization": "not-a-bearer-token"})
    assert response.status_code == 401


async def test_protected_route_with_valid_token_succeeds(real_auth_client):
    client, private_key = real_auth_client
    token = _valid_token(private_key)
    response = await client.get("/products", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200


async def test_health_is_public(real_auth_client):
    client, _ = real_auth_client
    response = await client.get("/health")
    assert response.status_code == 200
