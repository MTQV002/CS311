"""
RAG v3 - API Schemas (Full Original + Fixes)
============================================
Pydantic models for API request/response validation.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class IntentType(str, Enum):
    """Intent types for semantic routing"""
    CHAT = "CHAT"
    LAW = "LAW"


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================

class ChatRequest(BaseModel):
    """Chat request schema"""
    # ✅ FIX: Đổi 'message' -> 'content' để khớp với Frontend mới
    content: str = Field(
        ...,
        description="User's message/question",
        min_length=1,
        max_length=4000,
        examples=["Thời gian làm việc tối đa trong một tuần là bao nhiêu?"]
    )
    session_id: Optional[str] = Field(
        None,
        description="Session ID for conversation continuity",
        examples=["user-123-session-456"]
    )
    # Giữ lại các trường cũ cho tương thích ngược (dù có thể chưa dùng)
    skip_routing: bool = Field(
        False,
        description="Skip intent routing and always use RAG pipeline"
    )
    stream: bool = Field(
        True,
        description="Enable streaming response"
    )


class QueryRequest(BaseModel):
    """Simple query request (backward compatibility)"""
    question: str = Field(
        ...,
        description="User's question",
        min_length=1,
        max_length=4000
    )
    top_k: int = Field(
        5,
        description="Number of source documents to return",
        ge=1,
        le=20
    )


class ResetMemoryRequest(BaseModel):
    """Request to reset conversation history"""
    session_id: Optional[str] = None


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class SourceNode(BaseModel):
    """Structure for a retrieved document node"""
    text: str = Field(..., description="Content of the node")
    score: Optional[float] = Field(None, description="Similarity/Reranker score")
    id: Optional[str] = Field(None, description="Node ID")
    # ✅ FIX: Gom metadata vào dict để linh hoạt hơn
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Metadata (Article, Clause, Chapter...)"
    )


class ChatResponse(BaseModel):
    """Response standard (non-streaming)"""
    response: str
    sources: List[SourceNode] = []
    intent: Optional[str] = None


class QueryResponse(BaseModel):
    """Simple query response"""
    result: str
    source_nodes: List[SourceNode]


class ResetMemoryResponse(BaseModel):
    """Response for memory reset"""
    success: bool = Field(..., description="Whether reset was successful")
    message: str = Field(..., description="Status message")


class HealthResponse(BaseModel):
    """System health check response"""
    status: str
    version: str
    components: Dict[str, str]


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str
    message: str
    detail: Optional[str] = None