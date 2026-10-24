from httpx import ASGITransport, AsyncClient


async def test_metrics_is_public_and_in_prometheus_format(mongo_database):
    """Prometheus scrapes without a token, so /metrics must not sit behind the
    JWT dependency the product routes use."""
    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.get("/health")
        response = await client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert 'http_requests_total{handler="/health"' in response.text
    assert "http_request_duration_seconds" in response.text
