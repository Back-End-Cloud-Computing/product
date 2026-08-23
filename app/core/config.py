from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application configuration, sourced from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "product-service"
    app_env: str = "development"
    log_level: str = "INFO"

    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "product_service"

    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "products"

    embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embedding_device: str = "cpu"

    llm_provider: str = "mock"
    llm_model: str = "qwen/qwen3-8b:free"
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_timeout_seconds: int = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()
