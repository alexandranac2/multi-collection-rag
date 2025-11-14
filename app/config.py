"""Configuration management for the RAG API."""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    
    # LangFuse Configuration
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    
    # ChromaDB Configuration
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    
    # Document Storage
    DOCUMENTS_BASE_PATH: str = "./documents"
    
    # Q&A Collection Name
    QA_COLLECTION_NAME: str = "qa_history"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra environment variables


# Global settings instance
settings = Settings()

