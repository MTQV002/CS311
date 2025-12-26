"""
RAG v3 - API Routes (Full Original + SSE)
=========================================
FastAPI routes including Chat (SSE), Query (Legacy), and Utils.
"""
import json
import logging
from typing import List, Any
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse

from src.api.schemas import (
    ChatRequest, ChatResponse,
    QueryRequest, QueryResponse,
    ResetMemoryResponse,
    HealthResponse,
    SourceNode
)
from src.engine.chat_engine import (
    ChatEngineManager,
    get_chat_engine_manager
)
from src import __version__

# Setup logging
logger = logging.getLogger("uvicorn")

router = APIRouter()

# --- Helper Function (Giữ lại từ code cũ) ---
def _convert_source_nodes(source_nodes: List[Any]) -> List[SourceNode]:
    """Convert LlamaIndex nodes to API SourceNode schema"""
    result = []
    for node in source_nodes:
        # Xử lý an toàn cho Metadata
        meta = node.node.metadata or {}
        
        result.append(SourceNode(
            text=node.node.get_content()[:1000], # Cắt ngắn nếu quá dài
            score=node.score,
            id=node.node.node_id,
            metadata=meta
        ))
    return result


# ============================================================================
# CHAT ENDPOINT (SSE STREAMING)
# ============================================================================
@router.post("/chat")
async def chat(
    request: ChatRequest,
    engine: ChatEngineManager = Depends(get_chat_engine_manager)
):
    """
    Streaming Chat Endpoint (SSE).
    """
    async def event_generator():
        try:
            # Gọi hàm stream từ engine
            async for text, intent, nodes in engine.astream_chat(request.content):
                payload = {}
                
                # 1. Text Token
                if text:
                    payload["token"] = text
                
                # 2. Metadata (Intent & Sources)
                if intent:
                    # Chuyển enum sang string nếu cần
                    payload["intent"] = str(intent.value) if hasattr(intent, 'value') else str(intent)
                
                if nodes:
                    # Dùng helper function đã có để convert nodes
                    converted_nodes = _convert_source_nodes(nodes)
                    # Serialize pydantic models to dict
                    payload["nodes"] = [node.dict() for node in converted_nodes]
                
                if payload:
                    json_data = json.dumps(payload, ensure_ascii=False)
                    yield f"data: {json_data}\n\n"
                    
        except Exception as e:
            logger.error(f"Chat Error: {str(e)}")
            error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {error_data}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ============================================================================
# LEGACY QUERY ENDPOINT (Giữ lại cho bạn)
# ============================================================================
@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    engine: ChatEngineManager = Depends(get_chat_engine_manager)
):
    """
    Simple RAG Query (Non-conversational).
    Useful for testing retrieval quality directly.
    """
    try:
        # Sử dụng query engine thuần túy (không nhớ lịch sử)
        response = await engine.chat_engine.achat(request.question)
        
        return QueryResponse(
            result=str(response),
            source_nodes=_convert_source_nodes(response.source_nodes)
        )
    except Exception as e:
        logger.error(f"Query Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# UTILS & HEALTH
# ============================================================================
@router.post("/reset-memory", response_model=ResetMemoryResponse)
async def reset_memory(
    engine: ChatEngineManager = Depends(get_chat_engine_manager)
):
    """Reset conversation memory"""
    try:
        engine.reset()
        return ResetMemoryResponse(success=True, message="Memory reset successfully")
    except Exception as e:
        return ResetMemoryResponse(success=False, message=str(e))


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check"""
    try:
        engine = get_chat_engine_manager()
        status = "healthy" if engine._initialized else "initializing"
        
        components = {
            "llm": "ready" if engine.llm else "not loaded",
            "embedding": "ready" if engine.embed_model else "not loaded",
            "vector_store": "ready" if engine.vector_store else "not loaded",
            # Kiểm tra thêm các thành phần khác nếu cần
        }
        
        return HealthResponse(
            status=status,
            version=__version__,
            components=components
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            version=__version__,
            components={"error": str(e)}
        )