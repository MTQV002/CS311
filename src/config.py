"""Configuration settings for RAG v3 using Pydantic Settings"""
from typing import Optional, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )
    
    # ===== LLM Settings =====
    LLM_PROVIDER: Literal["gemini", "openai", "groq"] = "groq"
    GEMINI_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    
    # LLM Model Configuration
    LLM_MODEL_GEMINI: str = "gemini-2.5-flash-latest"
    LLM_MODEL_OPENAI: str = "gpt-4o-mini"
    LLM_MODEL_GROQ: str = "llama-3.3-70b-versatile"  # Options: llama-3.3-70b-versatile, llama-3.1-70b-versatile, mixtral-8x7b-32768
    LLM_TEMPERATURE: float = 0.1
    LLM_CONTEXT_WINDOW: int = 32768
    LLM_MAX_TOKENS: int = 2048
    
    # ===== Embedding Settings =====
    EMBEDDING_MODEL: str = "bkai-foundation-models/vietnamese-bi-encoder"
    EMBEDDING_DIM: int = 768
    EMBEDDING_BATCH_SIZE: int = 32
    
    # ===== Reranker Settings =====
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_TOP_N: int = 5
    HUGGINGFACE_API_KEY: Optional[str] = None  # Optional for HF Inference API
    
    # ===== Qdrant Cloud Settings =====
    QDRANT_URL: str = "http://localhost:6333"  # Qdrant Cloud URL
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "vietnam_labor_law_v3"
    
    # ===== Retrieval Settings =====
    VECTOR_TOP_K: int = 20
    BM25_TOP_K: int = 20
    HYBRID_TOP_K: int = 30  # After RRF fusion
    RRF_K: int = 60  # RRF constant
    
    # ===== Chunking Settings (for Ingestion) =====
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    
    # ===== Observability (Arize Phoenix) =====
    PHOENIX_COLLECTOR_ENDPOINT: Optional[str] = None
    ENABLE_TRACING: bool = False
    
    # ===== Server Settings =====
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # ===== Derived Properties =====
    @property
    def llm_model(self) -> str:
        """Get the appropriate LLM model based on provider"""
        if self.LLM_PROVIDER == "gemini":
            return self.LLM_MODEL_GEMINI
        elif self.LLM_PROVIDER == "groq":
            return self.LLM_MODEL_GROQ
        return self.LLM_MODEL_OPENAI
    
    @property
    def llm_api_key(self) -> Optional[str]:
        """Get the appropriate API key based on provider"""
        if self.LLM_PROVIDER == "gemini":
            return self.GEMINI_API_KEY
        elif self.LLM_PROVIDER == "groq":
            return self.GROQ_API_KEY
        return self.OPENAI_API_KEY


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()
