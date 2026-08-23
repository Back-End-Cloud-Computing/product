from abc import ABC, abstractmethod


class OrderServiceClient(ABC):
    """Interface for the future Order microservice integration."""

    @abstractmethod
    async def get_purchased_product_ids(self, user_id: str) -> list[str]:
        """Product ids the user has purchased in the past."""


class MockOrderServiceClient(OrderServiceClient):
    async def get_purchased_product_ids(self, user_id: str) -> list[str]:
        return []
