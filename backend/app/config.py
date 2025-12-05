"""
Centralized configuration management for the Document Q&A System.
All model configurations and system settings are defined here.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with validation."""
    
    # Database
    DATABASE_URL: str
    
    # Redis
    REDIS_URL: str
    
    # Qdrant
    QDRANT_URL: str
    QDRANT_API_KEY: str = ""
    
    # Cloudinary (PDF storage only)
    CLOUDINARY_CLOUD_NAME: str
    CLOUDINARY_API_KEY: str
    CLOUDINARY_API_SECRET: str
    
    # Model Configuration
    EMBEDDING_MODEL: str = "all-mpnet-base-v2"
    EMBEDDING_DIMENSION: int = 768
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    LLM_PROVIDER: str = "ollama"
    LLM_MODEL: str = "phi3.5"
    OLLAMA_URL: str = "http://localhost:11434/api/generate"
    
    # Chunking Configuration
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 150
    
    # Retrieval Configuration
    TOP_DOCS_DEFAULT: int = 8
    TOP_CHUNKS_PER_DOC_DEFAULT: int = 4
    
    # Cache TTL (seconds)
    CACHE_TTL_EMBEDDING: int = 86400  # 24 hours
    CACHE_TTL_RERANK: int = 3600      # 1 hour
    CACHE_TTL_ANSWER: int = 3600      # 1 hour
    
    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 10
    RATE_LIMIT_PER_HOUR: int = 100
    
    # Celery
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str
    
    # Application
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: str = "http://localhost:3000"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
