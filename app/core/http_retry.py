import httpx
import tenacity


def _is_transient_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


retry_transient_errors = tenacity.retry(
    retry=tenacity.retry_if_exception(_is_transient_error),
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=0.2, max=2),
    reraise=True,
)


@retry_transient_errors
async def post_with_retry(client: httpx.AsyncClient, url: str, json: dict) -> httpx.Response:
 
    response = await client.post(url, json=json)
    response.raise_for_status()
    return response
