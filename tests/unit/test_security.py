import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from app.core import security
from app.core.exceptions import AuthenticationError

USER_ID = "11111111-1111-1111-1111-111111111111"


@pytest.fixture
def keypair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


@pytest.fixture(autouse=True)
def _install_test_public_key(keypair, monkeypatch):
    _, public_key = keypair
    monkeypatch.setattr(security, "_public_key", public_key)
    yield


def _make_token(private_key, *, typ="access", issuer=security.ISSUER, exp_delta=3600, omit_claims=()):
    now = int(time.time())
    claims = {
        "iss": issuer,
        "sub": USER_ID,
        "email": "teste@ganjj.com",
        "role": "CLIENTE",
        "typ": typ,
        "iat": now,
        "exp": now + exp_delta,
    }
    for claim in omit_claims:
        claims.pop(claim, None)
    return jwt.encode(claims, private_key, algorithm="RS256")


async def test_valid_token_is_accepted(keypair):
    private_key, _ = keypair
    token = _make_token(private_key)

    user = await security.get_current_user(authorization=f"Bearer {token}")

    assert user == security.CurrentUser(id=USER_ID, email="teste@ganjj.com", role="CLIENTE")


async def test_missing_header_is_rejected():
    with pytest.raises(AuthenticationError):
        await security.get_current_user(authorization=None)


async def test_header_without_bearer_prefix_is_rejected():
    with pytest.raises(AuthenticationError):
        await security.get_current_user(authorization="Token abc123")


async def test_token_signed_by_another_key_is_rejected():
    other_private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    token = _make_token(other_private_key)

    with pytest.raises(AuthenticationError):
        await security.get_current_user(authorization=f"Bearer {token}")


async def test_expired_token_is_rejected(keypair):
    private_key, _ = keypair
    token = _make_token(private_key, exp_delta=-10)

    with pytest.raises(AuthenticationError):
        await security.get_current_user(authorization=f"Bearer {token}")


async def test_refresh_token_is_rejected_on_a_protected_route(keypair):
    private_key, _ = keypair
    token = _make_token(private_key, typ="refresh")

    with pytest.raises(AuthenticationError):
        await security.get_current_user(authorization=f"Bearer {token}")


async def test_wrong_issuer_is_rejected(keypair):
    private_key, _ = keypair
    token = _make_token(private_key, issuer="someone-else")

    with pytest.raises(AuthenticationError):
        await security.get_current_user(authorization=f"Bearer {token}")


async def test_token_missing_required_claim_is_rejected(keypair):
    private_key, _ = keypair
    token = _make_token(private_key, omit_claims=["role"])

    with pytest.raises(AuthenticationError):
        await security.get_current_user(authorization=f"Bearer {token}")


async def test_no_public_key_loaded_is_rejected(monkeypatch, keypair):
    private_key, _ = keypair
    monkeypatch.setattr(security, "_public_key", None)
    token = _make_token(private_key)

    with pytest.raises(AuthenticationError):
        await security.get_current_user(authorization=f"Bearer {token}")


async def test_valid_token_makes_auth_headers_forward_it(keypair):
    private_key, _ = keypair
    token = _make_token(private_key)

    await security.get_current_user(authorization=f"Bearer {token}")

    assert security.auth_headers() == {"Authorization": f"Bearer {token}"}


async def test_auth_headers_empty_without_a_validated_token():
    security._current_token.set(None)
    assert security.auth_headers() == {}
