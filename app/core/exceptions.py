class ProductServiceError(Exception):
    """Base exception for the product service domain."""


class ProductNotFoundError(ProductServiceError):
    def __init__(self, product_id: str):
        self.product_id = product_id
        super().__init__(f"Product '{product_id}' not found")


class DuplicateSkuError(ProductServiceError):
    def __init__(self, sku: str):
        self.sku = sku
        super().__init__(f"Product with SKU '{sku}' already exists")


class LLMProviderError(ProductServiceError):
    """Raised when the external LLM provider fails, times out, or is not configured."""


class EmbeddingGenerationError(ProductServiceError):
    """Raised when the embedding-reranking service fails to index a product or
    run a semantic search (embedding generation, or its call to vector-db)."""


class AgenticSearchError(ProductServiceError):
    """Raised when the agentic search loop fails to produce a result."""


class AuthenticationError(ProductServiceError):
    """Raised when the incoming request's bearer token is missing, malformed, or
    fails local verification against the authorization service's public key."""


class InsufficientStockError(ProductServiceError):
    def __init__(self, product_id: str, available_stock: int):
        self.product_id = product_id
        self.available_stock = available_stock
        super().__init__(f"Product '{product_id}' has only {available_stock} unit(s) in stock")


class StockOperationConflictError(ProductServiceError):
    """Raised when an Idempotency-Key is reused for a different stock request,
    or while the first request with that key is still in flight."""

    def __init__(self, operation_id: str):
        self.operation_id = operation_id
        super().__init__(f"Operation '{operation_id}' was already used for a different stock request")
