from typing import Optional, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
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
    LLM_MODEL_GROQ: str = "llama-3.3-70b-versatile" # openai/gpt-oss-120b,llama-3.1-8b-instant,openai/gpt-oss-20b,llama-3.3-70b-versatile
    LLM_TEMPERATURE: float = 0.05   
    LLM_CONTEXT_WINDOW: int = 131072  
    LLM_MAX_TOKENS: int = 4096      
    
    # ===== Embedding Settings =====
    EMBEDDING_MODEL: str = "AITeamVN/Vietnamese_Embedding"
    EMBEDDING_DIM: int = 1024
    EMBEDDING_BATCH_SIZE: int = 32
     
    # ===== Reranker Settings =====
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3" #    "BAAI/bge-reranker-v2-m3" # thanhtantran/Vietnamese_Reranker
    RERANKER_TOP_N: int = 7        
    HUGGINGFACE_API_KEY: Optional[str] = None
    
    # ===== Qdrant Cloud Settings =====
    QDRANT_URL: str = "http://localhost:6333" 
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "legal_decrees_LBV"
    
    # ===== Retrieval Settings =====
    VECTOR_TOP_K: int = 15          
    BM25_TOP_K: int = 15
    HYBRID_TOP_K: int = 15          
    RRF_K: int = 30                 
    
    # ===== Memory Settings =====
    MEMORY_TOKEN_LIMIT: int = 12000 
    
    # ===== Chunking Settings (for Ingestion) =====
    CHUNK_SIZE: int = 10000
    CHUNK_OVERLAP: int = 0
    
    # ===== Observability (Arize Phoenix) =====
    PHOENIX_COLLECTOR_ENDPOINT: Optional[str] = None
    ENABLE_TRACING: bool = False
    
    # ===== Server Settings =====
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # ===== Derived Properties =====
    @property
    def llm_model(self) -> str:
        if self.LLM_PROVIDER == "gemini":
            return self.LLM_MODEL_GEMINI
        elif self.LLM_PROVIDER == "groq":
            return self.LLM_MODEL_GROQ
        return self.LLM_MODEL_OPENAI
    
    @property
    def llm_api_key(self) -> Optional[str]:
        if self.LLM_PROVIDER == "gemini":
            return self.GEMINI_API_KEY
        elif self.LLM_PROVIDER == "groq":
            return self.GROQ_API_KEY
        return self.OPENAI_API_KEY


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
