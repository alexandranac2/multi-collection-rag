"""Configuration management for the RAG API."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, read from the environment / .env."""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # LangFuse (optional: tracing is off unless both keys are set)
    LANGFUSE_PUBLIC_KEY: str | None = None
    LANGFUSE_SECRET_KEY: str | None = None
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    # Storage
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    DOCUMENTS_BASE_PATH: str = "./documents"
    STATE_DIR: str = "./.rag_state"  # document registry + per-collection hash caches

    QA_COLLECTION_NAME: str = "qa_history"

    # Security
    API_KEY: str | None = None  # when set, every /api route requires X-API-Key
    CORS_ORIGINS: list[str] = ["*"]
    MAX_UPLOAD_MB: int = 25

    LOG_LEVEL: str = "INFO"


settings = Settings()
