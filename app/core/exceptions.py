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
