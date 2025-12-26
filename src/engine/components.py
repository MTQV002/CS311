"""
RAG v3 - Engine Components Factory
===================================
Factory functions for creating LLM, Embedding, and Reranker instances.
Uses LlamaIndex integrations for unified interface.
"""
from typing import Optional
from functools import lru_cache

from llama_index.core.llms import LLM
from llama_index.core.embeddings import BaseEmbedding
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from src.config import settings


@lru_cache()
def get_llm() -> LLM:
    """
    Get LLM instance based on configuration.
    
    Supports:
    - Groq (llama-3.3-70b-versatile) - FREE & FAST!
    - Google Gemini (gemini-1.5-flash)
    - OpenAI (gpt-4o-mini)
    
    Returns:
        LlamaIndex LLM instance
    """
    if settings.LLM_PROVIDER == "groq":
        from llama_index.llms.groq import Groq
        
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        
        print(f"🚀 Loading Groq LLM: {settings.llm_model}")
        llm = Groq(
            api_key=settings.GROQ_API_KEY,
            model=settings.llm_model,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            context_window=settings.LLM_CONTEXT_WINDOW,
        )
    elif settings.LLM_PROVIDER == "gemini":
        from llama_index.llms.gemini import Gemini
        
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        
        print(f"🤖 Loading Gemini LLM: {settings.llm_model}")
        llm = Gemini(
            api_key=settings.GEMINI_API_KEY,
            model=settings.llm_model,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
    else:  # openai
        from llama_index.llms.openai import OpenAI
        
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        
        print(f"🤖 Loading OpenAI LLM: {settings.llm_model}")
        llm = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=settings.llm_model,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
    
    print("✅ LLM loaded successfully")
    return llm


@lru_cache()
def get_embed_model() -> BaseEmbedding:
    """
    Get Vietnamese embedding model.
    
    Uses HuggingFace embedding with bkai-foundation-models/vietnamese-bi-encoder
    for optimal Vietnamese text understanding.
    
    Returns:
        LlamaIndex HuggingFaceEmbedding instance
    """
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    import os
    
    print(f"🔤 Loading embedding model: {settings.EMBEDDING_MODEL}")
    
    # Set cache folder for HuggingFace models to avoid permission issues
    cache_folder = os.path.expanduser("~/.cache/huggingface/hub")
    os.makedirs(cache_folder, exist_ok=True)
    
    embed_model = HuggingFaceEmbedding(
        model_name=settings.EMBEDDING_MODEL,
        embed_batch_size=settings.EMBEDDING_BATCH_SIZE,
        trust_remote_code=True,
        cache_folder=cache_folder,
    )
    print("✅ Embedding model loaded successfully")
    return embed_model


@lru_cache()
def get_reranker(top_n: Optional[int] = None) -> BaseNodePostprocessor:
    """
    Get reranker model for result refinement.
    
    Uses SentenceTransformerRerank (sbert) instead of FlagEmbedding
    for better stability.
    
    Args:
        top_n: Number of top results to return after reranking
        
    Returns:
        LlamaIndex node postprocessor for reranking
    """
    top_n = top_n or settings.RERANKER_TOP_N
    
    print(f"🎯 Loading reranker: {settings.RERANKER_MODEL}")
    
    # 👇 UPDATED CLASS: Use SentenceTransformerRerank
    reranker = SentenceTransformerRerank(
        model=settings.RERANKER_MODEL,
        top_n=top_n,
    )
    print("✅ Reranker loaded successfully")
    return reranker


def get_qdrant_client():
    """
    Get Qdrant client for vector store operations (Synchronous).
    
    Returns:
        QdrantClient instance
    """
    from qdrant_client import QdrantClient
    
    if settings.QDRANT_API_KEY:
        print(f"☁️  Connecting to Qdrant Cloud: {settings.QDRANT_URL}")
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
    else:
        print(f"🖥️  Connecting to local Qdrant: {settings.QDRANT_URL}")
        client = QdrantClient(url=settings.QDRANT_URL)
    
    print("✅ Qdrant client connected")
    return client


def get_vector_store(client=None, collection_name: Optional[str] = None):
    """
    Get Qdrant vector store instance.
    
    CRITICAL FIX: Initializes both Sync and Async clients to support 
    FastAPI/Chainlit async environment.
    
    Args:
        client: Optional QdrantClient instance
        collection_name: Collection name (default from settings)
        
    Returns:
        QdrantVectorStore instance
    """
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from qdrant_client import AsyncQdrantClient  # Import Async Client
    
    # 1. Get Sync Client
    if client is None:
        client = get_qdrant_client()
    
    # 2. Init Async Client (Required for Async Web Apps like Chainlit)
    if settings.QDRANT_API_KEY:
        aclient = AsyncQdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
    else:
        aclient = AsyncQdrantClient(url=settings.QDRANT_URL)

    collection_name = collection_name or settings.QDRANT_COLLECTION
    
    # 3. Pass BOTH clients to VectorStore
    vector_store = QdrantVectorStore(
        client=client,
        aclient=aclient,  # <--- Fixes "Async client is not initialized"
        collection_name=collection_name,
    )
    print(f"✅ Vector store ready: {collection_name}")
    return vector_store