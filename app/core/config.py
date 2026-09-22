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

    auth_service_base_url: str = "http://localhost:8081"

    embedding_reranking_base_url: str = "http://localhost:8003"
    embedding_reranking_timeout_seconds: int = 30

    vector_db_base_url: str = "http://localhost:8002"
    vector_db_timeout_seconds: int = 10

    llm_provider_base_url: str = "http://localhost:8004"
    llm_provider_timeout_seconds: int = 60

    agentic_search_max_iterations: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()
