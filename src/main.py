"""
RAG v3 - FastAPI Main Application
==================================
Production-grade FastAPI server with:
- Arize Phoenix observability/tracing
- Lifespan management for component initialization
- CORS middleware
- Error handling
"""
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import __version__
from src.config import settings
from src.api.routes import router
from src.engine.chat_engine import ChatEngineManager, set_chat_engine_manager


# ============================================================================
# PHOENIX TRACING SETUP
# ============================================================================

def setup_phoenix_tracing():
    """
    Setup Arize Phoenix for observability and tracing.
    
    Phoenix provides:
    - LLM call tracing
    - Retrieval performance metrics
    - Token usage tracking
    - Latency monitoring
    """
    if not settings.ENABLE_TRACING:
        print("📊 Tracing disabled")
        return
    
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
        
        # Setup tracer provider
        endpoint = settings.PHOENIX_COLLECTOR_ENDPOINT or "http://localhost:6006/v1/traces"
        
        tracer_provider = TracerProvider()
        trace.set_tracer_provider(tracer_provider)
        
        # Configure OTLP exporter
        otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
        span_processor = BatchSpanProcessor(otlp_exporter)
        tracer_provider.add_span_processor(span_processor)
        
        # Instrument LlamaIndex
        LlamaIndexInstrumentor().instrument()
        
        print(f"📊 Phoenix tracing enabled: {endpoint}")
        
    except ImportError as e:
        print(f"⚠️ Phoenix tracing not available: {e}")
        print("   Install with: pip install arize-phoenix openinference-instrumentation-llama-index")
    except Exception as e:
        print(f"⚠️ Phoenix tracing setup failed: {e}")


# ============================================================================
# NODE LOADING FOR BM25
# ============================================================================

def load_nodes_from_qdrant() -> List:
    """
    Load all nodes from Qdrant for BM25 index.
    
    This fetches document content from Qdrant to build
    the in-memory BM25 index on startup.
    """
    from qdrant_client import QdrantClient
    from llama_index.core.schema import TextNode
    
    print("📚 Loading nodes from Qdrant for BM25 index...")
    
    try:
        # Connect to Qdrant
        if settings.QDRANT_API_KEY:
            client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
            )
        else:
            client = QdrantClient(url=settings.QDRANT_URL)
        
        # Check collection exists
        collections = [c.name for c in client.get_collections().collections]
        if settings.QDRANT_COLLECTION not in collections:
            print(f"⚠️ Collection '{settings.QDRANT_COLLECTION}' not found")
            print("   Run: python scripts/ingest.py to create it")
            return []
        
        # Scroll through all points
        points = []
        offset = None
        batch_size = 100
        
        while True:
            result = client.scroll(
                collection_name=settings.QDRANT_COLLECTION,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False  # Don't need vectors for BM25
            )
            
            batch_points, offset = result
            points.extend(batch_points)
            
            if offset is None:
                break
        
        print(f"📄 Loaded {len(points)} points from Qdrant")
        
        # Convert to TextNodes
        nodes = []
        for i, point in enumerate(points):
            payload = point.payload or {}
            
            # Extract text (try different field names)
            text = payload.get('text') or payload.get('_node_content') or ""
            
            # If _node_content is JSON, parse it
            if isinstance(text, str) and text.startswith('{'):
                try:
                    import json
                    content_data = json.loads(text)
                    text = content_data.get('text', text)
                except:
                    pass
            
            if not text:
                continue
            
            # Build metadata
            metadata = {
                'article': payload.get('article'),
                'clause': payload.get('clause'),
                'chapter': payload.get('chapter'),
                'chapter_title': payload.get('chapter_title'),
                'section': payload.get('section'),
                'article_title': payload.get('article_title'),
                'source': payload.get('source', 'Vietnam Labor Law 2019'),
            }
            
            node = TextNode(
                text=text,
                metadata=metadata,
                id_=str(point.id) if point.id else f"node_{i}"
            )
            nodes.append(node)
        
        print(f"✅ Created {len(nodes)} TextNodes for BM25")
        return nodes
        
    except Exception as e:
        print(f"⚠️ Failed to load nodes from Qdrant: {e}")
        print("   The system will run with vector-only retrieval")
        return []


# ============================================================================
# FASTAPI LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler for startup/shutdown events.
    
    Startup:
    - Setup Phoenix tracing
    - Load nodes from Qdrant
    - Initialize ChatEngineManager
    
    Shutdown:
    - Cleanup resources
    """
    print("=" * 60)
    print("🚀 RAG v3 - Vietnam Labor Law Assistant")
    print(f"   Version: {__version__}")
    print("=" * 60)
    
    # Startup
    print("\n📦 Starting up...")
    
    # 1. Setup tracing
    setup_phoenix_tracing()
    
    # 2. Load nodes from Qdrant
    nodes = load_nodes_from_qdrant()
    
    # 3. Initialize chat engine
    print("\n🔧 Initializing Chat Engine...")
    engine = ChatEngineManager(nodes=nodes)
    engine.initialize()
    set_chat_engine_manager(engine)
    
    print("\n✅ Server ready!")
    print(f"   API: http://{settings.API_HOST}:{settings.API_PORT}")
    print(f"   Docs: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    print("=" * 60)
    
    yield  # Application runs here
    
    # Shutdown
    print("\n👋 Shutting down...")
    print("✅ Cleanup complete")


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Vietnam Labor Law RAG v3",
    description="""
🏛️ **Production-Grade Vietnam Labor Law QA System**

Powered by LlamaIndex with:
- 🤖 Semantic Intent Routing (CHAT vs LAW)
- 💬 Conversational Memory (CondensePlusContextChatEngine)
- 🔍 Hybrid Search (Vector + BM25 + RRF)
- 🎯 BGE Reranker
- 📊 Arize Phoenix Observability

**Endpoints:**
- `POST /chat` - Main chat with routing
- `POST /query` - Simple query (v2 compatible)
- `POST /reset-memory` - Reset conversation
- `GET /health` - Health check
    """,
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routes
app.include_router(router)


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect to docs"""
    return {
        "name": "Vietnam Labor Law RAG v3",
        "version": __version__,
        "docs": "/docs",
        "health": "/health"
    }


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,  # Disable in production
        log_level="info"
    )
