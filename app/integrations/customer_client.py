from abc import ABC, abstractmethod


class CustomerServiceClient(ABC):
    """Interface for the future Customer microservice integration.

    Covers browsing/search behavior and cross-user similarity, which in this
    architecture are owned by the Customer domain rather than by Product.
    """

    @abstractmethod
    async def get_viewed_product_ids(self, user_id: str) -> list[str]:
        """Product ids recently viewed by the user."""

    @abstractmethod
    async def get_recent_search_terms(self, user_id: str) -> list[str]:
        """The user's most recent search queries."""

    @abstractmethod
    async def get_similar_users_purchases(self, user_id: str) -> list[str]:
        """Product ids purchased by users with similar behavior/profile."""


class MockCustomerServiceClient(CustomerServiceClient):
    async def get_viewed_product_ids(self, user_id: str) -> list[str]:
        return []

    async def get_recent_search_terms(self, user_id: str) -> list[str]:
        return []

    async def get_similar_users_purchases(self, user_id: str) -> list[str]:
        return []
