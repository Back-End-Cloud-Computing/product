from app.integrations.cart_client import CartServiceClient, MockCartServiceClient
from app.integrations.customer_client import CustomerServiceClient, MockCustomerServiceClient
from app.integrations.order_client import MockOrderServiceClient, OrderServiceClient

__all__ = [
    "CartServiceClient",
    "CustomerServiceClient",
    "OrderServiceClient",
    "get_cart_client",
    "get_order_client",
    "get_customer_client",
]


def get_cart_client() -> CartServiceClient:
    """Factory for the Cart client. Swap the mock for a real HTTP client once
    the Cart microservice is available, gated by configuration."""
    return MockCartServiceClient()


def get_order_client() -> OrderServiceClient:
    return MockOrderServiceClient()


def get_customer_client() -> CustomerServiceClient:
    return MockCustomerServiceClient()
