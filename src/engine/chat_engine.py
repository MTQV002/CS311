"""
RAG v3 - Chat Engine with Semantic Router
==========================================
Production-grade chat engine featuring:
- Semantic Intent Router (CHAT vs LAW)
- CondensePlusContextChatEngine for conversational RAG
- Query Rewriting for follow-up questions
- Full streaming support
"""
from typing import List, Optional, AsyncGenerator, Tuple
from enum import Enum
from dataclasses import dataclass

from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.base.response.schema import StreamingResponse

from src.config import settings
from src.engine.components import get_llm, get_embed_model, get_reranker, get_vector_store
from src.engine.retriever import HybridRetriever, HybridRetrieverFactory


class IntentType(str, Enum):
    """Intent types for semantic routing"""
    CHAT = "CHAT"   # General conversation
    LAW = "LAW"     # Legal questions requiring RAG


@dataclass
class RouterResult:
    """Result from semantic router"""
    intent: IntentType
    confidence: float
    reasoning: str


# ============================================================================
# PROMPTS
# ============================================================================

ROUTER_PROMPT = """Bạn là bộ phân loại ý định (intent classifier). Nhiệm vụ của bạn là xác định xem câu hỏi của người dùng thuộc loại nào.

Có 2 loại ý định:
1. **LAW**: Câu hỏi về pháp luật lao động Việt Nam, quyền và nghĩa vụ người lao động, hợp đồng lao động, tiền lương, thời gian làm việc, sa thải, bảo hiểm xã hội, v.v.
2. **CHAT**: Chào hỏi, hỏi thăm, câu hỏi chung không liên quan đến luật lao động, hoặc yêu cầu giải thích về hệ thống.

Trả lời CHÍNH XÁC theo định dạng:
INTENT: [LAW hoặc CHAT]
CONFIDENCE: [0.0-1.0]
REASONING: [Giải thích ngắn gọn tại sao]

Câu hỏi: {query}

Phân loại:"""

CONDENSE_PROMPT = """Cho lịch sử hội thoại và câu hỏi tiếp theo của người dùng, hãy viết lại câu hỏi thành một câu hỏi độc lập, đầy đủ ngữ cảnh.

Lịch sử hội thoại:
{chat_history}

Câu hỏi tiếp theo: {question}

Nếu câu hỏi tiếp theo đã đầy đủ ngữ cảnh, hãy giữ nguyên.
Nếu câu hỏi dùng đại từ (nó, họ, điều đó...) hoặc thiếu ngữ cảnh, hãy viết lại cho rõ ràng.

Câu hỏi đã viết lại:"""

CONTEXT_PROMPT = """Bạn là trợ lý AI chuyên gia về Luật Lao động Việt Nam 2019. Nhiệm vụ của bạn là trả lời câu hỏi dựa trên các điều khoản pháp luật được cung cấp.

NGUYÊN TẮC TRẢ LỜI:
1. Chỉ dựa vào các điều khoản được cung cấp bên dưới
2. Trích dẫn chính xác số Điều, Khoản khi trả lời (ví dụ: "Theo Điều 5, Khoản 2...")
3. Trả lời bằng tiếng Việt, rõ ràng, ngắn gọn và dễ hiểu
4. Nếu không tìm thấy thông tin chính xác, hãy nói rõ điều đó
5. Không bịa đặt hoặc suy luận ngoài nội dung được cung cấp

CÁC ĐIỀU KHOẢN LIÊN QUAN:
{context_str}

Hãy trả lời câu hỏi một cách chính xác và hữu ích."""

CHAT_RESPONSE_PROMPT = """Bạn là trợ lý AI thân thiện về Luật Lao động Việt Nam. Hãy trả lời câu hỏi chung hoặc chào hỏi của người dùng một cách thân thiện và chuyên nghiệp.

Nếu người dùng hỏi về khả năng của bạn, hãy giải thích rằng bạn có thể:
- Trả lời câu hỏi về Luật Lao động Việt Nam 2019
- Tra cứu các điều khoản về quyền và nghĩa vụ người lao động
- Giải thích về hợp đồng lao động, tiền lương, thời gian làm việc
- Tư vấn về các quy định sa thải, bảo hiểm xã hội

Câu hỏi: {query}

Trả lời (bằng tiếng Việt):"""


class SemanticRouter:
    """
    Semantic Router for intent classification.
    
    Uses LLM to classify user queries into:
    - CHAT: General conversation (respond directly)
    - LAW: Legal questions (trigger RAG pipeline)
    """
    
    def __init__(self, llm=None):
        self.llm = llm or get_llm()
    
    def route(self, query: str) -> RouterResult:
        """
        Classify query intent.
        
        Args:
            query: User's question
            
        Returns:
            RouterResult with intent type, confidence, and reasoning
        """
        prompt = ROUTER_PROMPT.format(query=query)
        
        response = self.llm.complete(prompt)
        response_text = str(response)
        
        # Parse response
        try:
            lines = response_text.strip().split('\n')
            intent_line = next((l for l in lines if l.startswith('INTENT:')), None)
            confidence_line = next((l for l in lines if l.startswith('CONFIDENCE:')), None)
            reasoning_line = next((l for l in lines if l.startswith('REASONING:')), None)
            
            intent_str = intent_line.split(':')[1].strip().upper() if intent_line else "LAW"
            confidence = float(confidence_line.split(':')[1].strip()) if confidence_line else 0.8
            reasoning = reasoning_line.split(':', 1)[1].strip() if reasoning_line else "Classified based on query content"
            
            intent = IntentType.LAW if intent_str == "LAW" else IntentType.CHAT
            
        except Exception as e:
            print(f"⚠️ Router parsing error: {e}, defaulting to LAW")
            intent = IntentType.LAW
            confidence = 0.5
            reasoning = f"Parsing error, defaulting to LAW: {str(e)}"
        
        return RouterResult(intent=intent, confidence=confidence, reasoning=reasoning)
    
    async def aroute(self, query: str) -> RouterResult:
        """Async version of route"""
        prompt = ROUTER_PROMPT.format(query=query)
        
        response = await self.llm.acomplete(prompt)
        response_text = str(response)
        
        # Parse response (same logic as sync)
        try:
            lines = response_text.strip().split('\n')
            intent_line = next((l for l in lines if l.startswith('INTENT:')), None)
            confidence_line = next((l for l in lines if l.startswith('CONFIDENCE:')), None)
            reasoning_line = next((l for l in lines if l.startswith('REASONING:')), None)
            
            intent_str = intent_line.split(':')[1].strip().upper() if intent_line else "LAW"
            confidence = float(confidence_line.split(':')[1].strip()) if confidence_line else 0.8
            reasoning = reasoning_line.split(':', 1)[1].strip() if reasoning_line else "Classified based on query content"
            
            intent = IntentType.LAW if intent_str == "LAW" else IntentType.CHAT
            
        except Exception as e:
            print(f"⚠️ Router parsing error: {e}, defaulting to LAW")
            intent = IntentType.LAW
            confidence = 0.5
            reasoning = f"Parsing error, defaulting to LAW: {str(e)}"
        
        return RouterResult(intent=intent, confidence=confidence, reasoning=reasoning)


class ChatEngineManager:
    """
    Main Chat Engine Manager for RAG v3.
    
    Features:
    - Semantic routing (CHAT vs LAW intents)
    - CondensePlusContextChatEngine for conversational RAG
    - Hybrid retrieval (Vector + BM25)
    - Reranking for result refinement
    - Streaming support
    """
    
    def __init__(
        self,
        nodes: Optional[List[TextNode]] = None,
        memory_token_limit: int = 4096
    ):
        """
        Initialize ChatEngineManager.
        
        Args:
            nodes: Document nodes for BM25 retriever
            memory_token_limit: Max tokens for conversation memory
        """
        self.nodes = nodes or []
        self.memory_token_limit = memory_token_limit
        
        self.llm = None
        self.embed_model = None
        self.reranker = None
        self.router = None
        self.hybrid_retriever = None
        self.chat_engine = None
        self.memory = None
        
        self._initialized = False
    
    def initialize(self, nodes: Optional[List[TextNode]] = None):
        """
        Initialize all components.
        
        Args:
            nodes: Optional document nodes (overrides constructor nodes)
        """
        if nodes:
            self.nodes = nodes
        
        print("🚀 Initializing Chat Engine Manager...")
        
        # 1. Initialize core components
        print("[1/6] Loading LLM...")
        self.llm = get_llm()
        
        print("[2/6] Loading embedding model...")
        self.embed_model = get_embed_model()
        
        print("[3/6] Loading reranker...")
        self.reranker = get_reranker()
        
        # 2. Initialize router
        print("[4/6] Initializing semantic router...")
        self.router = SemanticRouter(self.llm)
        
        # 3. Create vector store index
        print("[5/6] Creating vector index and hybrid retriever...")
        vector_store = get_vector_store()
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=self.embed_model,
        )
        
        # 4. Create hybrid retriever
        if self.nodes:
            self.hybrid_retriever = HybridRetrieverFactory.create_from_index(
                index=index,
                nodes=self.nodes,
            )
        else:
            # If no nodes provided, use vector-only retriever
            print("⚠️ No nodes provided, using vector-only retrieval")
            self.hybrid_retriever = index.as_retriever(
                similarity_top_k=settings.VECTOR_TOP_K
            )
        
        # 5. Create chat engine with memory
        print("[6/6] Creating chat engine with memory...")
        self.memory = ChatMemoryBuffer.from_defaults(
            token_limit=self.memory_token_limit
        )
        
        self.chat_engine = CondensePlusContextChatEngine.from_defaults(
            retriever=self.hybrid_retriever,
            llm=self.llm,
            memory=self.memory,
            node_postprocessors=[self.reranker] if self.reranker else None,
            context_prompt=CONTEXT_PROMPT,
            condense_prompt=CONDENSE_PROMPT,
            verbose=True,
        )
        
        self._initialized = True
        print("✅ Chat Engine Manager initialized successfully!")
    
    def _ensure_initialized(self):
        """Ensure engine is initialized"""
        if not self._initialized:
            raise RuntimeError("ChatEngineManager not initialized. Call initialize() first.")
    
    def reset_memory(self):
        """Reset conversation memory"""
        self._ensure_initialized()
        self.memory.reset()
        print("🔄 Conversation memory reset")
    
    def get_chat_history(self) -> List[ChatMessage]:
        """Get current chat history"""
        self._ensure_initialized()
        return self.memory.get_all()
    
    def _handle_chat_intent(self, query: str) -> str:
        """
        Handle CHAT intent (general conversation).
        
        Args:
            query: User's question
            
        Returns:
            LLM response for general conversation
        """
        try:
            prompt = CHAT_RESPONSE_PROMPT.format(query=query)
            response = self.llm.complete(prompt)
            # Extract text from response (handle both .text attribute and string conversion)
            if hasattr(response, 'text'):
                return response.text or ""
            return str(response) if response else ""
        except Exception as e:
            print(f"❌ LLM Error in _handle_chat_intent: {e}")
            return "Xin lỗi, tôi gặp lỗi khi kết nối với dịch vụ AI. Vui lòng thử lại sau."
    
    async def _ahandle_chat_intent(self, query: str) -> str:
        """Async version of _handle_chat_intent"""
        try:
            prompt = CHAT_RESPONSE_PROMPT.format(query=query)
            response = await self.llm.acomplete(prompt)
            # Extract text from response (handle both .text attribute and string conversion)
            if hasattr(response, 'text'):
                return response.text or ""
            return str(response) if response else ""
        except Exception as e:
            print(f"❌ LLM Error in _ahandle_chat_intent: {e}")
            return "Xin lỗi, tôi gặp lỗi khi kết nối với dịch vụ AI. Vui lòng thử lại sau."
    
    def chat(
        self,
        query: str,
        skip_routing: bool = False
    ) -> Tuple[str, IntentType, List[NodeWithScore]]:
        """
        Process a chat message.
        
        Args:
            query: User's question
            skip_routing: If True, always use RAG pipeline
            
        Returns:
            Tuple of (response_text, intent_type, source_nodes)
        """
        self._ensure_initialized()
        
        source_nodes = []
        
        # Step 1: Route intent (unless skipped)
        if skip_routing:
            intent = IntentType.LAW
            print(f"⏭️ Routing skipped, using LAW intent")
        else:
            router_result = self.router.route(query)
            intent = router_result.intent
            print(f"🎯 Router: {intent.value} (confidence: {router_result.confidence:.2f})")
        
        # Step 2: Handle based on intent
        if intent == IntentType.CHAT:
            response_text = self._handle_chat_intent(query)
            print(f"💬 CHAT response_text type: {type(response_text)}, value: {repr(response_text)}")
            # Add to memory for context
            self.memory.put(ChatMessage(role=MessageRole.USER, content=query))
            self.memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=response_text or ""))
        else:
            # Use RAG chat engine
            response = self.chat_engine.chat(query)
            response_text = str(response) if response else ""
            print(f"⚖️ LAW response_text type: {type(response_text)}, value: {repr(response_text[:100] if response_text else 'None')}")
            source_nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
        
        # Ensure response_text is never None
        response_text = response_text or "Xin lỗi, tôi không thể tạo câu trả lời lúc này."
        print(f"✅ Final response_text type: {type(response_text)}, length: {len(response_text)}")
        return response_text, intent, source_nodes
    
    async def achat(
        self,
        query: str,
        skip_routing: bool = False
    ) -> Tuple[str, IntentType, List[NodeWithScore]]:
        """
        Async version of chat.
        
        Args:
            query: User's question
            skip_routing: If True, always use RAG pipeline
            
        Returns:
            Tuple of (response_text, intent_type, source_nodes)
        """
        self._ensure_initialized()
        
        source_nodes = []
        
        # Step 1: Route intent
        if skip_routing:
            intent = IntentType.LAW
            print(f"⏭️ Routing skipped, using LAW intent")
        else:
            router_result = await self.router.aroute(query)
            intent = router_result.intent
            print(f"🎯 Router: {intent.value} (confidence: {router_result.confidence:.2f})")
        
        # Step 2: Handle based on intent
        if intent == IntentType.CHAT:
            response_text = await self._ahandle_chat_intent(query)
            print(f"💬 CHAT response_text type: {type(response_text)}, value: {repr(response_text)}")
            self.memory.put(ChatMessage(role=MessageRole.USER, content=query))
            self.memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=response_text or ""))
        else:
            response = await self.chat_engine.achat(query)
            response_text = str(response) if response else ""
            print(f"⚖️ LAW response_text type: {type(response_text)}, value: {repr(response_text[:100] if response_text else 'None')}")
            source_nodes = response.source_nodes if hasattr(response, 'source_nodes') else []
        
        # Ensure response_text is never None
        response_text = response_text or "Xin lỗi, tôi không thể tạo câu trả lời lúc này."
        print(f"✅ Final response_text type: {type(response_text)}, length: {len(response_text)}")
        return response_text, intent, source_nodes
    
    async def astream_chat(
        self,
        query: str,
        skip_routing: bool = False
    ) -> AsyncGenerator[Tuple[str, Optional[IntentType], Optional[List[NodeWithScore]]], None]:
        """
        Stream chat response asynchronously.
        
        Yields chunks of response text, then final metadata.
        
        Args:
            query: User's question
            skip_routing: If True, always use RAG pipeline
            
        Yields:
            Tuple of (text_chunk, intent_type, source_nodes)
            - During streaming: (chunk, None, None)
            - Final yield: ("", intent, source_nodes)
        """
        self._ensure_initialized()
        
        # Step 1: Route intent
        if skip_routing:
            intent = IntentType.LAW
        else:
            router_result = await self.router.aroute(query)
            intent = router_result.intent
            print(f"🎯 Router: {intent.value} (confidence: {router_result.confidence:.2f})")
        
        # Step 2: Handle based on intent
        if intent == IntentType.CHAT:
            # For CHAT, stream from LLM directly
            prompt = CHAT_RESPONSE_PROMPT.format(query=query)
            full_response = ""
            
            async for chunk in await self.llm.astream_complete(prompt):
                chunk_text = chunk.delta if hasattr(chunk, 'delta') else str(chunk)
                full_response += chunk_text
                yield chunk_text, None, None
            
            # Add to memory
            self.memory.put(ChatMessage(role=MessageRole.USER, content=query))
            self.memory.put(ChatMessage(role=MessageRole.ASSISTANT, content=full_response))
            
            # Final yield with metadata
            yield "", intent, []
        else:
            # Use RAG chat engine streaming
            streaming_response = await self.chat_engine.astream_chat(query)
            source_nodes = []
            
            async for chunk in streaming_response.async_response_gen():
                yield chunk, None, None
            
            # Get source nodes after streaming
            if hasattr(streaming_response, 'source_nodes'):
                source_nodes = streaming_response.source_nodes
            
            # Final yield with metadata
            yield "", intent, source_nodes


# Singleton instance for FastAPI lifespan
_chat_engine_manager: Optional[ChatEngineManager] = None


def get_chat_engine_manager() -> ChatEngineManager:
    """Get or create ChatEngineManager singleton"""
    global _chat_engine_manager
    if _chat_engine_manager is None:
        _chat_engine_manager = ChatEngineManager()
    return _chat_engine_manager


def set_chat_engine_manager(manager: ChatEngineManager):
    """Set ChatEngineManager singleton"""
    global _chat_engine_manager
    _chat_engine_manager = manager
