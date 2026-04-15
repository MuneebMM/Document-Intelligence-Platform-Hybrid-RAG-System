"""Application configuration loaded from environment variables via pydantic-settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the Document Intelligence Platform.

    All values are read from environment variables (case-insensitive) or a
    .env file at the project root.  Secret fields have no default so that a
    missing value raises a clear ValidationError at startup.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Secrets (required) ---
    openai_api_key: str
    cohere_api_key: str

    # --- Qdrant ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "documents"

    # --- App ---
    environment: str = "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application Settings instance.

    Using lru_cache ensures the .env file is read exactly once for the
    lifetime of the process, making repeated calls free.
    """
    return Settings()


settings: Settings = get_settings()
