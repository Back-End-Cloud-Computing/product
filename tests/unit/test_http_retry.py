import httpx
import pytest
import respx

from app.core.http_retry import post_with_retry


async def test_post_with_retry_retries_on_5xx_then_succeeds():
    with respx.mock(base_url="http://svc") as mock:
        route = mock.post("/x").mock(side_effect=[httpx.Response(503), httpx.Response(200, json={"ok": True})])
        async with httpx.AsyncClient(base_url="http://svc") as client:
            response = await post_with_retry(client, "/x", {})

    assert response.json() == {"ok": True}
    assert route.call_count == 2


async def test_post_with_retry_retries_on_connection_error_then_succeeds():
    with respx.mock(base_url="http://svc") as mock:
        route = mock.post("/x").mock(
            side_effect=[httpx.ConnectError("boom"), httpx.Response(200, json={"ok": True})]
        )
        async with httpx.AsyncClient(base_url="http://svc") as client:
            response = await post_with_retry(client, "/x", {})

    assert response.json() == {"ok": True}
    assert route.call_count == 2


async def test_post_with_retry_does_not_retry_on_4xx():
    with respx.mock(base_url="http://svc") as mock:
        route = mock.post("/x").mock(return_value=httpx.Response(422))
        async with httpx.AsyncClient(base_url="http://svc") as client:
            with pytest.raises(httpx.HTTPStatusError):
                await post_with_retry(client, "/x", {})

    assert route.call_count == 1


async def test_post_with_retry_gives_up_after_3_attempts():
    with respx.mock(base_url="http://svc") as mock:
        route = mock.post("/x").mock(return_value=httpx.Response(500))
        async with httpx.AsyncClient(base_url="http://svc") as client:
            with pytest.raises(httpx.HTTPStatusError):
                await post_with_retry(client, "/x", {})

    assert route.call_count == 3
