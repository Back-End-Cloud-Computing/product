from abc import ABC, abstractmethod


class CartServiceClient(ABC):
    """Interface for the future Cart microservice integration.

    A real implementation (e.g. an HTTP client) can be swapped in later via
    `get_cart_client()` without touching `recommendation_service`.
    """

    @abstractmethod
    async def get_cart_product_ids(self, user_id: str) -> list[str]:
        """Product ids currently in the user's cart."""


class MockCartServiceClient(CartServiceClient):
    async def get_cart_product_ids(self, user_id: str) -> list[str]:
        return []
